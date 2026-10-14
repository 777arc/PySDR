import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

# Figures for the SDR Frameworks chapter

####################################################
# Anatomy of a framework, using the FM receiver
####################################################
fig, ax = plt.subplots(figsize=(14.5, 4.6))
ax.set_xlim(0, 14.5)
ax.set_ylim(0, 4.6)
ax.axis('off')

block_color = '#cfe2f3'
block_edge = '#3d6d99'
buffer_color = '#f6d7a7'
sched_color = '#d9ead3'

block_w, block_h, block_y = 2.3, 1.3, 2.6
block_xs = [0.3, 4.1, 7.9, 11.7]
block_names = ['File Source', 'FM Demod', 'Low-Pass Filter\n+ Decimate by 5', 'WAV File Sink']
for x, name in zip(block_xs, block_names):
    ax.add_patch(FancyBboxPatch((x, block_y), block_w, block_h, boxstyle='round,pad=0.05', fc=block_color, ec=block_edge, lw=2))
    ax.text(x + block_w/2, block_y + block_h/2, name, ha='center', va='center', fontsize=11.5, fontweight='bold')
    # stream ports, little squares on the left/right edges
    if name != 'File Source':
        ax.add_patch(Rectangle((x - 0.12, block_y + block_h/2 - 0.12), 0.24, 0.24, fc='white', ec=block_edge, lw=1.5, zorder=3))
    if name != 'WAV File Sink':
        ax.add_patch(Rectangle((x + block_w - 0.12, block_y + block_h/2 - 0.12), 0.24, 0.24, fc='white', ec=block_edge, lw=1.5, zorder=3))

# Buffers between blocks, drawn as a row of cells, with the sample rate/type of each connection underneath
rates = ['complex\n250 kHz', 'real\n250 kHz', 'real\n50 kHz']
for i in range(3):
    x_start = block_xs[i] + block_w + 0.12
    x_stop = block_xs[i + 1] - 0.12
    n_cells = 6
    cell_w = (x_stop - x_start - 0.2) / n_cells
    y = block_y + block_h/2 - 0.15
    for c in range(n_cells):
        filled = c < 4  # some cells full of samples, some empty
        ax.add_patch(Rectangle((x_start + 0.1 + c*cell_w, y), cell_w, 0.3, fc=buffer_color if filled else 'white', ec='#9c6b1e', lw=1))
    ax.annotate('', xy=(x_stop, block_y + block_h/2 + 0.32), xytext=(x_start, block_y + block_h/2 + 0.32), arrowprops=dict(arrowstyle='->', lw=1.5, color='k'))
    ax.text((x_start + x_stop)/2, block_y + block_h/2 + 0.45, rates[i], ha='center', va='bottom', fontsize=10, style='italic', linespacing=1.1)

# Scheduler underneath, with dashed lines up to each block
ax.add_patch(FancyBboxPatch((0.3, 0.45), 13.7, 0.75, boxstyle='round,pad=0.05', fc=sched_color, ec='#4f7a3a', lw=2))
ax.text(7.15, 0.825, 'Scheduler: decides when each block runs, how many samples it gets, and on which thread', ha='center', va='center', fontsize=12)
for x in block_xs:
    ax.plot([x + block_w/2, x + block_w/2], [1.25, block_y - 0.07], ls=':', color='#4f7a3a', lw=1.5)

# Labels for the parts
ax.annotate('port', xy=(block_xs[1] - 0.12, block_y + block_h/2 - 0.14), xytext=(block_xs[1] - 0.12, block_y - 0.35), ha='center', va='top', fontsize=11, color=block_edge, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=block_edge, lw=1.2))
buf_x = (block_xs[1] + block_w + block_xs[2])/2
ax.annotate('buffer', xy=(buf_x, block_y + block_h/2 - 0.17), xytext=(buf_x, block_y - 0.35), ha='center', va='top', fontsize=11, color='#9c6b1e', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#9c6b1e', lw=1.2))

fig.savefig('../_images/sdr_frameworks_anatomy.svg', bbox_inches='tight')


####################################################
# Timeline
####################################################
events = [
    (2000, 'SCA 1.0\n(JTRS)', 'gov'),
    (2001, 'GNU Radio', 'oss'),
    (2010, 'OpenCPI', 'gov'),
    (2012, 'REDHAWK', 'gov'),
    (2021, 'FutureSDR', 'oss'),
    (2021, 'CyberEther', 'oss'),
    (2026, 'GNU Radio 4.0\n(release candidates)', 'oss'),
]
colors = {'oss': '#3d6d99', 'gov': '#a33a3a'}

fig, ax = plt.subplots(figsize=(14, 3.6))
ax.set_xlim(1998, 2028)
ax.set_ylim(-2.2, 2.2)
ax.axhline(0, color='k', lw=1.5)
for year in range(2000, 2028, 5):
    ax.plot([year, year], [-0.08, 0.08], color='k', lw=1.5)
    ax.text(year, -0.3, str(year), ha='center', va='top', fontsize=11)

heights = [1.2, -1.3, 1.2, -1.3, 1.2, -1.3, 1.2]  # alternate above/below the line so labels don't collide
for (year, label, kind), h in zip(events, heights):
    ax.plot([year, year], [0, h], color=colors[kind], lw=1.5)
    ax.plot(year, 0, 'o', color=colors[kind], markersize=8, zorder=3)
    ax.text(year, h + (0.08 if h > 0 else -0.08), label, ha='center', va='bottom' if h > 0 else 'top', fontsize=11, color=colors[kind], fontweight='bold')

ax.plot([], [], 'o', color=colors['oss'], label='Community / open source origin')
ax.plot([], [], 'o', color=colors['gov'], label='Government / defense origin')
ax.legend(loc='upper left', bbox_to_anchor=(0.12, 1.0), fontsize=10, frameon=False)
ax.axis('off')

fig.savefig('../_images/sdr_frameworks_timeline.svg', bbox_inches='tight')

plt.show()
