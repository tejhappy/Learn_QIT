import multiprocessing as mp
import numpy as np
from qutip import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _unitary_evolution(initial_state, t):
    theta = np.pi * t
    U = (-1j * theta * sigmay()).expm()
    return U * initial_state


def _irreversible_evolution(initial_state, t):
    gamma = min(max(0.7 * t, 0.0), 1.0)
    K0 = Qobj([[1, 0], [0, np.sqrt(1 - gamma)]])
    K1 = Qobj([[0, np.sqrt(gamma)], [0, 0]])
    rho_in = initial_state * initial_state.dag()
    rho_out = K0 * rho_in * K0.dag() + K1 * rho_in * K1.dag()
    eigenvalues, eigenvectors = rho_out.eigenstates()
    state = eigenvectors[0] if eigenvalues[0] > eigenvalues[1] else eigenvectors[1]
    return state, rho_out


def _compute_frame_data(t, initial_state_array):
    initial_state = Qobj(initial_state_array, dims=[[2], [1]])

    state_u = _unitary_evolution(initial_state, t)
    rho_u = state_u * state_u.dag()
    purity_u = np.real((rho_u * rho_u).tr())
    prob0_u = np.real(rho_u[0, 0])
    prob1_u = np.real(rho_u[1, 1])

    state_d, rho_d = _irreversible_evolution(initial_state, t)
    purity_d = np.real((rho_d * rho_d).tr())
    prob0_d = np.real(rho_d[0, 0])
    prob1_d = np.real(rho_d[1, 1])

    data_u = {
        'state': state_u,
        'rho': rho_u,
        'purity': purity_u,
        'prob0': prob0_u,
        'prob1': prob1_u
    }
    data_d = {
        'state': state_d,
        'rho': rho_d,
        'purity': purity_d,
        'prob0': prob0_d,
        'prob1': prob1_d
    }

    return data_u, data_d

class QuantumChannelVisualizer:
    def __init__(self, initial_state=None):
        """
        Initialize with custom initial state (default: |+⟩)
        """
        if initial_state is None:
            self.initial_state = (basis(2, 0) + basis(2, 1)).unit()
        else:
            self.initial_state = initial_state
        
        print(f"✨ Initial State: |ψ₀⟩ = {self.initial_state[0,0]:.3f}|0⟩ + {self.initial_state[1,0]:.3f}|1⟩")
    
    def unitary_evolution(self, t):
        """Unitary evolution (reversible)"""
        theta = np.pi * t
        U = (-1j * theta * sigmay()).expm()
        return U * self.initial_state
    
    def irreversible_evolution(self, t):
        """Amplitude damping channel (irreversible)"""
        gamma = min(max(0.7 * t, 0.0), 1.0)
        K0 = Qobj([[1, 0], [0, np.sqrt(1 - gamma)]])
        K1 = Qobj([[0, np.sqrt(gamma)], [0, 0]])
        rho_in = self.initial_state * self.initial_state.dag()
        rho_out = K0 * rho_in * K0.dag() + K1 * rho_in * K1.dag()
        
        eigenvalues, eigenvectors = rho_out.eigenstates()
        return eigenvectors[0] if eigenvalues[0] > eigenvalues[1] else eigenvectors[1], rho_out
    
    def create_animation(self, filename='quantum_channel.html', use_multiprocessing=True):
        """Create interactive HTML animation"""
        print("📊 Generating interactive HTML visualization...")
        
        frames_count = 100
        time_points = np.linspace(0, 1, frames_count)
        
        # Pre-compute all data
        initial_state_array = self.initial_state.full()
        results = []
        if use_multiprocessing:
            try:
                ctx = mp.get_context("spawn")
                with ctx.Pool() as pool:
                    results = pool.starmap(
                        _compute_frame_data,
                        [(t, initial_state_array) for t in time_points]
                    )
            except Exception as exc:
                print(f"⚠️  Multiprocessing failed, falling back to single core: {exc}")
                results = [_compute_frame_data(t, initial_state_array) for t in time_points]
        else:
            results = [_compute_frame_data(t, initial_state_array) for t in time_points]

        data_u = [result[0] for result in results]
        data_d = [result[1] for result in results]
        
        # Create figure
        fig = make_subplots(
            rows=3, cols=4,
            subplot_titles=(
                '<b>Unitary: Density Matrix ρ</b>', '<b>Unitary: Complex Amplitudes</b>',
                '<b>Damping: Density Matrix ρ</b>', '<b>Damping: Complex Amplitudes</b>',
                '<b>Unitary: Probabilities</b>', '<b>Unitary: Purity</b>',
                '<b>Damping: Probabilities</b>', '<b>Damping: Purity</b>',
                '<b>Unitary: State Vector</b>', '',
                '<b>Damping: State Vector</b>', ''
            ),
            specs=[
                [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}, {"type": "xy"}],
                [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}, {"type": "xy"}],
                [{"type": "xy", "colspan": 2}, None, {"type": "xy", "colspan": 2}, None]
            ],
            row_heights=[0.36, 0.22, 0.42],
            vertical_spacing=0.18,
            horizontal_spacing=0.10
        )
        
        # Initial frame
        du = data_u[0]
        dd = data_d[0]
        
        # Row 1, Col 1: Unitary Density Matrix
        matrix_u = du['rho'].full()
        fig.add_trace(go.Heatmap(
            z=np.abs(matrix_u),
            zmin=0,
            zmax=1,
            zsmooth=False,
            x=['|0⟩', '|1⟩'],
            y=['⟨0|', '⟨1|'],
            colorscale='Plasma',
            showscale=False,
            xgap=3,
            ygap=3,
            text=[[f"{matrix_u[i,j].real:.2f}{matrix_u[i,j].imag:+.2f}i" 
                   for j in range(2)] for i in range(2)],
            texttemplate='%{text}',
            textfont=dict(size=14, color='white'),
            hovertemplate='%{y}, %{x}<br>|Value|: %{z:.3f}<extra></extra>'
        ), row=1, col=1)
        
        # Row 1, Col 2: Unitary Complex Plane
        theta_vals = np.linspace(0, 2*np.pi, 100)
        fig.add_trace(go.Scatter(
            x=np.cos(theta_vals), y=np.sin(theta_vals),
            mode='lines',
            line=dict(color='rgba(0,212,255,0.3)', width=2, dash='dash'),
            showlegend=False,
            hoverinfo='skip'
        ), row=1, col=2)
        
        alpha_u = complex(du['state'][0, 0])
        beta_u = complex(du['state'][1, 0])
        
        fig.add_trace(go.Scatter(
            x=[0, alpha_u.real],
            y=[0, alpha_u.imag],
            mode='lines+markers',
            line=dict(color='#00ff88', width=5),
            marker=dict(size=[0, 20], color='#00ff88', symbol='arrow-bar-up'),
            showlegend=False,
            hovertemplate=f'α: {alpha_u.real:.3f}{alpha_u.imag:+.3f}i<extra></extra>'
        ), row=1, col=2)
        
        fig.add_trace(go.Scatter(
            x=[0, beta_u.real],
            y=[0, beta_u.imag],
            mode='lines+markers',
            line=dict(color='cyan', width=5),
            marker=dict(size=[0, 20], color='cyan', symbol='arrow-bar-up'),
            showlegend=False,
            hovertemplate=f'β: {beta_u.real:.3f}{beta_u.imag:+.3f}i<extra></extra>'
        ), row=1, col=2)
        
        # Row 1, Col 3: Damping Density Matrix
        matrix_d = dd['rho'].full()
        fig.add_trace(go.Heatmap(
            z=np.abs(matrix_d),
            zmin=0,
            zmax=1,
            zsmooth=False,
            x=['|0⟩', '|1⟩'],
            y=['⟨0|', '⟨1|'],
            colorscale='Plasma',
            showscale=False,
            xgap=3,
            ygap=3,
            text=[[f"{matrix_d[i,j].real:.2f}{matrix_d[i,j].imag:+.2f}i" 
                   for j in range(2)] for i in range(2)],
            texttemplate='%{text}',
            textfont=dict(size=14, color='white'),
            hovertemplate='%{y}, %{x}<br>|Value|: %{z:.3f}<extra></extra>'
        ), row=1, col=3)
        
        # Row 1, Col 4: Damping Complex Plane
        fig.add_trace(go.Scatter(
            x=np.cos(theta_vals), y=np.sin(theta_vals),
            mode='lines',
            line=dict(color='rgba(0,212,255,0.3)', width=2, dash='dash'),
            showlegend=False,
            hoverinfo='skip'
        ), row=1, col=4)
        
        alpha_d = complex(dd['state'][0, 0])
        beta_d = complex(dd['state'][1, 0])
        
        fig.add_trace(go.Scatter(
            x=[0, alpha_d.real],
            y=[0, alpha_d.imag],
            mode='lines+markers',
            line=dict(color='#ff4444', width=5),
            marker=dict(size=[0, 20], color='#ff4444', symbol='arrow-bar-up'),
            showlegend=False,
            hovertemplate=f'α: {alpha_d.real:.3f}{alpha_d.imag:+.3f}i<extra></extra>'
        ), row=1, col=4)
        
        fig.add_trace(go.Scatter(
            x=[0, beta_d.real],
            y=[0, beta_d.imag],
            mode='lines+markers',
            line=dict(color='cyan', width=5),
            marker=dict(size=[0, 20], color='cyan', symbol='arrow-bar-up'),
            showlegend=False,
            hovertemplate=f'β: {beta_d.real:.3f}{beta_d.imag:+.3f}i<extra></extra>'
        ), row=1, col=4)
        
        # Row 2: Probabilities and Purity
        times_init = time_points[:1]
        
        fig.add_trace(go.Scatter(
            x=times_init, y=[du['prob0']],
            mode='lines+markers',
            name='P(|0⟩)',
            line=dict(color='#ff00ff', width=3),
            marker=dict(size=6),
            showlegend=False
        ), row=2, col=1)
        
        fig.add_trace(go.Scatter(
            x=times_init, y=[du['prob1']],
            mode='lines+markers',
            name='P(|1⟩)',
            line=dict(color='#ffff00', width=3),
            marker=dict(size=6),
            showlegend=False
        ), row=2, col=1)
        
        fig.add_trace(go.Scatter(
            x=times_init, y=[du['purity']],
            mode='lines+markers',
            line=dict(color='#00ff88', width=4),
            marker=dict(size=6),
            fill='tozeroy',
            fillcolor='rgba(0,255,136,0.2)',
            showlegend=False
        ), row=2, col=2)
        
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[1, 1],
            mode='lines',
            line=dict(color='green', width=2, dash='dash'),
            showlegend=False,
            hoverinfo='skip'
        ), row=2, col=2)
        
        fig.add_trace(go.Scatter(
            x=times_init, y=[dd['prob0']],
            mode='lines+markers',
            line=dict(color='#ff00ff', width=3),
            marker=dict(size=6),
            showlegend=False
        ), row=2, col=3)
        
        fig.add_trace(go.Scatter(
            x=times_init, y=[dd['prob1']],
            mode='lines+markers',
            line=dict(color='#ffff00', width=3),
            marker=dict(size=6),
            showlegend=False
        ), row=2, col=3)
        
        fig.add_trace(go.Scatter(
            x=times_init, y=[dd['purity']],
            mode='lines+markers',
            line=dict(color='#ff4444', width=4),
            marker=dict(size=6),
            fill='tozeroy',
            fillcolor='rgba(255,68,68,0.2)',
            showlegend=False
        ), row=2, col=4)
        
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[1, 1],
            mode='lines',
            line=dict(color='green', width=2, dash='dash'),
            showlegend=False,
            hoverinfo='skip'
        ), row=2, col=4)
        
        # Row 3: State vectors
        alpha_u_abs = abs(alpha_u)
        beta_u_abs = abs(beta_u)
        phase_alpha_u = np.degrees(np.angle(alpha_u))
        phase_beta_u = np.degrees(np.angle(beta_u))
        
        fig.add_trace(go.Bar(
            x=['|α|', '|β|', 'φ(α)°', 'φ(β)°'],
            y=[alpha_u_abs, beta_u_abs, phase_alpha_u/360, phase_beta_u/360],
            marker=dict(color=['#ff00ff', 'cyan', '#88ff00', '#88ff00']),
            text=[f'{alpha_u_abs:.3f}', f'{beta_u_abs:.3f}', 
                  f'{phase_alpha_u:.1f}°', f'{phase_beta_u:.1f}°'],
            textposition='outside',
            showlegend=False
        ), row=3, col=1)
        
        alpha_d_abs = abs(alpha_d)
        beta_d_abs = abs(beta_d)
        phase_alpha_d = np.degrees(np.angle(alpha_d))
        phase_beta_d = np.degrees(np.angle(beta_d))
        
        fig.add_trace(go.Bar(
            x=['|α|', '|β|', 'φ(α)°', 'φ(β)°'],
            y=[alpha_d_abs, beta_d_abs, phase_alpha_d/360, phase_beta_d/360],
            marker=dict(color=['#ff00ff', 'cyan', '#88ff00', '#88ff00']),
            text=[f'{alpha_d_abs:.3f}', f'{beta_d_abs:.3f}', 
                  f'{phase_alpha_d:.1f}°', f'{phase_beta_d:.1f}°'],
            textposition='outside',
            showlegend=False
        ), row=3, col=3)
        
        # Create animation frames
        frames = []
        for idx in range(frames_count):
            t = time_points[idx]
            du = data_u[idx]
            dd = data_d[idx]
            
            # Update all trace data
            times_so_far = time_points[:idx+1]
            probs0_u = [data_u[i]['prob0'] for i in range(idx+1)]
            probs1_u = [data_u[i]['prob1'] for i in range(idx+1)]
            purity_u_vals = [data_u[i]['purity'] for i in range(idx+1)]
            probs0_d = [data_d[i]['prob0'] for i in range(idx+1)]
            probs1_d = [data_d[i]['prob1'] for i in range(idx+1)]
            purity_d_vals = [data_d[i]['purity'] for i in range(idx+1)]
            
            # Update density matrices
            matrix_u = du['rho'].full()
            matrix_d = dd['rho'].full()
            
            # Update complex amplitudes
            alpha_u = complex(du['state'][0, 0])
            beta_u = complex(du['state'][1, 0])
            alpha_d = complex(dd['state'][0, 0])
            beta_d = complex(dd['state'][1, 0])
            
            # Update state vectors
            alpha_u_abs = abs(alpha_u)
            beta_u_abs = abs(beta_u)
            phase_alpha_u = np.degrees(np.angle(alpha_u))
            phase_beta_u = np.degrees(np.angle(beta_u))
            alpha_d_abs = abs(alpha_d)
            beta_d_abs = abs(beta_d)
            phase_alpha_d = np.degrees(np.angle(alpha_d))
            phase_beta_d = np.degrees(np.angle(beta_d))
            
            # Update text labels
            text_u = [[f"{matrix_u[i,j].real:.2f}{matrix_u[i,j].imag:+.2f}i" for j in range(2)] for i in range(2)]
            text_d = [[f"{matrix_d[i,j].real:.2f}{matrix_d[i,j].imag:+.2f}i" for j in range(2)] for i in range(2)]
            text_bar_u = [f'{alpha_u_abs:.3f}', f'{beta_u_abs:.3f}', f'{phase_alpha_u:.1f}°', f'{phase_beta_u:.1f}°']
            text_bar_d = [f'{alpha_d_abs:.3f}', f'{beta_d_abs:.3f}', f'{phase_alpha_d:.1f}°', f'{phase_beta_d:.1f}°']

            frames.append(go.Frame(
                data=[
                    go.Heatmap(
                        z=np.abs(matrix_u),
                        zmin=0,
                        zmax=1,
                        zsmooth=False,
                        x=['|0⟩', '|1⟩'],
                        y=['⟨0|', '⟨1|'],
                        colorscale='Plasma',
                        showscale=False,
                        xgap=3,
                        ygap=3,
                        text=text_u,
                        texttemplate='%{text}',
                        textfont=dict(size=14, color='white'),
                        hovertemplate='%{y}, %{x}<br>|Value|: %{z:.3f}<extra></extra>'
                    ),
                    go.Scatter(x=[0, alpha_u.real], y=[0, alpha_u.imag]),
                    go.Scatter(x=[0, beta_u.real], y=[0, beta_u.imag]),
                    go.Heatmap(
                        z=np.abs(matrix_d),
                        zmin=0,
                        zmax=1,
                        zsmooth=False,
                        x=['|0⟩', '|1⟩'],
                        y=['⟨0|', '⟨1|'],
                        colorscale='Plasma',
                        showscale=False,
                        xgap=3,
                        ygap=3,
                        text=text_d,
                        texttemplate='%{text}',
                        textfont=dict(size=14, color='white'),
                        hovertemplate='%{y}, %{x}<br>|Value|: %{z:.3f}<extra></extra>'
                    ),
                    go.Scatter(x=[0, alpha_d.real], y=[0, alpha_d.imag]),
                    go.Scatter(x=[0, beta_d.real], y=[0, beta_d.imag]),
                    go.Scatter(x=times_so_far, y=probs0_u),
                    go.Scatter(x=times_so_far, y=probs1_u),
                    go.Scatter(x=times_so_far, y=purity_u_vals),
                    go.Scatter(x=times_so_far, y=probs0_d),
                    go.Scatter(x=times_so_far, y=probs1_d),
                    go.Scatter(x=times_so_far, y=purity_d_vals),
                    go.Bar(
                        x=['|α|', '|β|', 'φ(α)°', 'φ(β)°'],
                        y=[alpha_u_abs, beta_u_abs, phase_alpha_u/360, phase_beta_u/360],
                        marker=dict(color=['#ff00ff', 'cyan', '#88ff00', '#88ff00']),
                        text=text_bar_u,
                        textposition='outside'
                    ),
                    go.Bar(
                        x=['|α|', '|β|', 'φ(α)°', 'φ(β)°'],
                        y=[alpha_d_abs, beta_d_abs, phase_alpha_d/360, phase_beta_d/360],
                        marker=dict(color=['#ff00ff', 'cyan', '#88ff00', '#88ff00']),
                        text=text_bar_d,
                        textposition='outside'
                    )
                ],
                name=str(idx),
                traces=[0, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 16, 17]
            ))
        
        # Update layout
        fig.update_layout(
            height=1650,
            margin=dict(t=120, b=120, l=50, r=50),
            title=dict(
                text="<b style='color:#00d4ff'>🌟 QUANTUM CHANNELS 🌟</b><br>" +
                     "<span style='font-size:16px; color:#00ff88'>🔄 Unitary (Reversible)</span>  vs  " +
                     "<span style='font-size:16px; color:#ff4444'>💀 Amplitude Damping (Irreversible)</span>",
                x=0.5,
                y=0.98,
                xanchor='center',
                yanchor='top',
                font=dict(size=22, family='Arial Black')
            ),
            template='plotly_dark',
            paper_bgcolor='#0a0a0a',
            plot_bgcolor='#1a1a2e',
            showlegend=False,
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'buttons': [
                    {'label': '▶ Play', 'method': 'animate',
                     'args': [None, {'frame': {'duration': 50, 'redraw': True},
                                    'fromcurrent': True, 'mode': 'immediate'}]},
                    {'label': '⏸ Pause', 'method': 'animate',
                     'args': [[None], {'frame': {'duration': 0, 'redraw': False},
                                      'mode': 'immediate'}]}
                ],
                'x': 0.1, 'y': -0.08
            }],
            sliders=[{
                'active': 0,
                'steps': [{'args': [[f.name], {'frame': {'duration': 0, 'redraw': True},
                                               'mode': 'immediate'}],
                          'label': f't={int(f.name)/(frames_count - 1):.2f}',
                          'method': 'animate'} for f in frames],
                'x': 0.1, 'len': 0.85,
                'xanchor': 'left', 'y': -0.05
            }]
        )
        
        fig.frames = frames
        
        # Update axes
        fig.update_xaxes(showgrid=True, gridcolor='rgba(74,74,106,0.3)', showline=True, linecolor='#00d4ff')
        fig.update_yaxes(showgrid=True, gridcolor='rgba(74,74,106,0.3)', showline=True, linecolor='#00d4ff')
        
        # Fix heatmap axes - lock dimensions
        fig.update_xaxes(range=[-0.5, 1.5], row=1, col=1)
        fig.update_yaxes(range=[1.5, -0.5], row=1, col=1)
        fig.update_xaxes(range=[-0.5, 1.5], row=1, col=3)
        fig.update_yaxes(range=[1.5, -0.5], row=1, col=3)
        
        # State vector ranges
        fig.update_yaxes(range=[0, 1.05], row=3, col=1)
        fig.update_yaxes(range=[0, 1.05], row=3, col=3)
        
        # Complex plane aspect ratio
        fig.update_xaxes(range=[-1.2, 1.2], row=1, col=2)
        fig.update_yaxes(range=[-1.2, 1.2], row=1, col=2)
        fig.update_xaxes(range=[-1.2, 1.2], row=1, col=4)
        fig.update_yaxes(range=[-1.2, 1.2], row=1, col=4)
        
        # Save
        fig.write_html(filename, auto_open=True)

# Example usage
if __name__ == "__main__":
    mp.freeze_support()
    print("=" * 60)
    print("🌌 QUANTUM CHANNEL VISUALIZER 🌌")
    print("=" * 60)
    
    # Using |+⟩ state
    viz = QuantumChannelVisualizer()
    viz.create_animation('quantum_channel_plus.html')
    
    print("\n✨ Done! Check your browser for the interactive visualization.")
