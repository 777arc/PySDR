"""
Recreate sine-wave.png and impulse1.png with German labels.
Outputs go directly to _images_de/.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import os

OUT_DIR = os.path.join(os.path.dirname(__file__), '../_images_de')

# ── shared style helpers ──────────────────────────────────────────────────────

def arrow_axis(ax, xlabel, arrow_len_x=1.08, arrow_len_y=1.08, fontsize=13):
    """Draw clean arrow-tip axes, hide the default spines/ticks."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    # x-axis arrow
    ax.annotate('', xy=(arrow_len_x, 0), xycoords=('axes fraction', 'data'),
                xytext=(0, 0), textcoords=('axes fraction', 'data'),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.2))
    # y-axis arrow
    ax.annotate('', xy=(0, arrow_len_y), xycoords=('data', 'axes fraction'),
                xytext=(0, 0), textcoords=('data', 'axes fraction'),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.2))

    ax.text(arrow_len_x + 0.01, 0, xlabel,
            transform=ax.get_yaxis_transform(),
            fontsize=fontsize, va='center', ha='left',
            fontstyle='italic')


# ── sine-wave.png (German: Zeit / Frequenz) ───────────────────────────────────

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.2))
fig.subplots_adjust(wspace=0.35, left=0.04, right=0.96, top=0.88, bottom=0.12)

# Left: cosine wave — ~1.5 cycles, dashed tail, light grid
t = np.linspace(0, 1.3, 500)
y = np.cos(2 * np.pi * 1.15 * t)
split = 380
ax1.plot(t[:split], y[:split], color='#1a6fdb', lw=2.0, solid_capstyle='round')
ax1.plot(t[split-1:], y[split-1:], color='#1a6fdb', lw=2.0, linestyle='--')
ax1.set_xlim(-0.05, 1.55)
ax1.set_ylim(-1.5, 1.7)
ax1.grid(True, color='#cccccc', lw=0.6, zorder=0)
arrow_axis(ax1, 'Zeit', arrow_len_x=1.06, arrow_len_y=1.12)
ax1.text(0.5, -0.22, r'$\cos(2\pi f t)$', transform=ax1.transAxes,
         fontsize=12, ha='center', va='top', fontstyle='italic')

# Right: frequency impulse at f — solid line (no arrowhead on spike)
ax2.plot([0.38, 0.38], [0.0, 0.75], color='#1a6fdb', lw=2.5,
         solid_capstyle='round', transform=ax2.get_xaxis_transform())
ax2.set_xlim(-0.05, 1.0)
ax2.set_ylim(-0.1, 1.1)
ax2.text(0.38, -0.12, r'$f$', transform=ax2.get_xaxis_transform(),
         fontsize=12, ha='center', va='top', fontstyle='italic')
arrow_axis(ax2, 'Frequenz', arrow_len_x=1.06, arrow_len_y=1.12)

fig.savefig(os.path.join(OUT_DIR, 'sine-wave.png'), dpi=150, bbox_inches='tight',
            facecolor='white')
plt.close(fig)
print('Saved sine-wave.png')


# ── impulse1.png (German: Zeit / Frequenz) ────────────────────────────────────

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.2))
fig.subplots_adjust(wspace=0.35, left=0.04, right=0.96, top=0.88, bottom=0.12)

# Left: single impulse in time domain
ax1.annotate('', xy=(0.35, 0.82), xycoords=('data', 'axes fraction'),
             xytext=(0.35, 0.0), textcoords=('data', 'axes fraction'),
             arrowprops=dict(arrowstyle='->', color='#1a6fdb', lw=3.5,
                             mutation_scale=12))
ax1.set_xlim(-0.05, 1.0)
ax1.set_ylim(-0.1, 1.1)
arrow_axis(ax1, 'Zeit', arrow_len_x=1.06, arrow_len_y=1.12)

# Right: flat magnitude (solid + dotted tail)
x_flat = np.linspace(0.0, 0.72, 200)
ax2.plot(x_flat, np.ones_like(x_flat) * 0.45, color='#1a6fdb', lw=2.8,
         solid_capstyle='round')
x_dot = np.linspace(0.72, 0.98, 100)
ax2.plot(x_dot, np.ones_like(x_dot) * 0.45, color='#1a6fdb', lw=2.8,
         linestyle=':', solid_capstyle='round')
ax2.set_xlim(-0.05, 1.0)
ax2.set_ylim(-0.1, 1.1)
arrow_axis(ax2, 'Frequenz', arrow_len_x=1.06, arrow_len_y=1.12)

fig.savefig(os.path.join(OUT_DIR, 'impulse1.png'), dpi=150, bbox_inches='tight',
            facecolor='white')
plt.close(fig)
print('Saved impulse1.png')


# ── dc-signal.png / dc-signal1.png (German: Zeit / Frequenz) ─────────────────

def make_dc_signal():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.2))
    fig.subplots_adjust(wspace=0.35, left=0.04, right=0.96, top=0.88, bottom=0.12)

    # Left: flat DC line in time domain
    ax1.plot([0.05, 0.72], [0.45, 0.45], color='blue', lw=3.5,
             solid_capstyle='round', transform=ax1.get_xaxis_transform())
    ax1.set_xlim(-0.05, 1.0)
    ax1.set_ylim(-0.1, 1.1)
    arrow_axis(ax1, 'Zeit', arrow_len_x=1.06, arrow_len_y=1.12)

    # Right: impulse at 0 (DC component)
    ax2.plot([0.0, 0.0], [0.0, 0.78], color='blue', lw=3.5,
             solid_capstyle='round', transform=ax2.get_xaxis_transform())
    ax2.set_xlim(-0.15, 1.0)
    ax2.set_ylim(-0.1, 1.1)
    ax2.text(0.0, -0.12, '0', transform=ax2.get_xaxis_transform(),
             fontsize=12, ha='center', va='top')
    arrow_axis(ax2, 'Frequenz', arrow_len_x=1.06, arrow_len_y=1.12)

    return fig

fig = make_dc_signal()
fig.savefig(os.path.join(OUT_DIR, 'dc-signal.png'), dpi=150, bbox_inches='tight',
            facecolor='white')
plt.close(fig)
print('Saved dc-signal.png')

fig = make_dc_signal()
fig.savefig(os.path.join(OUT_DIR, 'dc-signal1.png'), dpi=150, bbox_inches='tight',
            facecolor='white')
plt.close(fig)
print('Saved dc-signal1.png')


# ── symbols1.png (German: DATEN / Zeit / Ein Symbol / 7 Symbole gesamt) ──────

bits = [1, 0, 1, 0, 0, 1, 0]
n = len(bits)

# Build sharp square-wave path with a short lead-in at LOW level
x_wave = [-0.25, 0]
y_wave = [0, 0]
for i, b in enumerate(bits):
    if i == 0:
        x_wave += [0, 0]
        y_wave += [0, b]
    else:
        x_wave += [i, i]
        y_wave += [bits[i-1], b]
    x_wave.append(i + 1)
    y_wave.append(b)

fig, ax = plt.subplots(figsize=(10, 2.5))
for spine in ax.spines.values():
    spine.set_visible(False)
ax.set_xticks([])
ax.set_yticks([])

ax.plot(x_wave, y_wave, color='#1f2fa8', lw=2.5, solid_capstyle='butt')

# Bit labels centred in each symbol slot
for i, b in enumerate(bits):
    ax.text(i + 0.5, 1.18, str(b), ha='center', va='bottom',
            fontsize=15, color='#1f2fa8')

# DATEN label on the far left, vertically centred on the wave
ax.text(-0.55, 0.5, 'DATEN', ha='right', va='center',
        fontsize=16, fontweight='bold', color='black')

# "(7 Symbole gesamt)" in red italic to the right of the wave
ax.text(n + 0.15, 0.5, '(7 Symbole gesamt)', ha='left', va='center',
        fontsize=14, color='red', fontstyle='italic')

# "Zeit →" small label + arrow at lower right of waveform
ax.annotate('', xy=(n - 0.05, -0.22), xytext=(n - 1.55, -0.22),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.3))
ax.text(n - 1.62, -0.22, 'Zeit', ha='right', va='center',
        fontsize=12, color='black')

# Red measurement bracket under the 3rd symbol (index 2, the second "1")
bx_l, bx_r = 2.0, 3.0
bx_mid = (bx_l + bx_r) / 2
by = -0.42
tick = 0.09
ax.plot([bx_l, bx_r], [by, by], color='red', lw=2.0, solid_capstyle='butt')
ax.plot([bx_l, bx_l], [by - tick, by + tick], color='red', lw=2.0)
ax.plot([bx_r, bx_r], [by - tick, by + tick], color='red', lw=2.0)
ax.text(bx_mid, by - 0.14, 'Ein Symbol', ha='center', va='top',
        fontsize=14, color='red', fontweight='bold')

ax.set_xlim(-1.0, n + 3.2)
ax.set_ylim(-1.1, 1.7)

fig.savefig(os.path.join(OUT_DIR, 'symbols1.png'), dpi=150, bbox_inches='tight',
            facecolor='white')
plt.close(fig)
print('Saved symbols1.png')
