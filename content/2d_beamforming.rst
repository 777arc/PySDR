.. _2d-beamforming-chapter:

##############
2D Beamforming
##############

***********************
Interactive Design Tool
***********************

The following interactive tool was created by Jason Durbin, a free-lancing phased array engineer, who graciously allowed it to be embedded within PySDR; feel free to visit the `full project <https://jasondurbin.github.io/PhasedArrayVisualizer>`_ or his `consulting business <https://neonphysics.com/>`_.  This tool allows you to change a phased array's geometry, element spacing, steering position, add sidelobe tapering, and other features.

Antenna elements are assumed to be isotropic. However, the directivity calculation assumes half-hemisphere radiation (e.g. no back lobes). Therefore, the computed directivity will be 3 dBi higher than using pure isotropic. Said in different terms, the individual element gain is +3.0 dBi. The mesh can be made finer by increasing theta/phi, u/v, or az/el points. However, increasing the number of points may result in laggy performance. Clicking (or long pressing) elements in the phase/attenuation plots allows you to manually set phase/attenuation. <b>Be sure to select "enable override."</b> Additionally, the attenuation pop-up allows you to disable elements. Hovering (or touching) the 2-D farfield plot or geometry plots will show the value of the plot under the cursor.

.. raw:: html

	<input type="text" id="pa-atten-manual" hidden />
	<input type="text" id="pa-phase-manual" hidden />
	<div class="text-group">
		<div class="pa-settings">
			<div id="pa-geometry-controls">
				<h3>Geometry</h3>
			</div>
			<div>
				<h3>Steering</h3>
				<select id="pa-steering-domain" style="width:100%;"></select>
				<div class="form-group" id="pa-theta-div">
					<label for="pa-theta">Theta (deg)</label>
					<input type="number" min="-90" max="90" value="0" id="pa-theta" name="pa-theta" />
				</div>
				<div class="form-group" id="pa-phi-div">
					<label for="pa-phi">Phi (deg)</label>
					<input type="number" min="-90" max="90" value="0" id="pa-phi" name="pa-phi" />
				</div>
			</div>
			<div>
				<h3>Taper(s)</h3>
				<div class="form-group" id="pa-taper-sampling-div">
					<label for="pa-taper-sampling">Sampling</label>
					<select id="pa-taper-sampling"><option>X & Y</option><option>Radial</option></select>
				</div>
				<div id="pa-taper-x-group" style="margin: 5px 0px;"></div>
				<div id="pa-taper-y-group" style="margin: 5px 0px;"></div>
			</div>
			<div>
				<h3>Quantization</h3>
				<div class="form-group" id="pa-phase-bits-div">
					<label for="pa-phase-bits">Phase Bits</label>
					<input type="number" min="0" max="10" value="0" step="1" id="pa-phase-bits" name="pa-phase-bits" />
				</div>
				<div class="form-group" id="pa-atten-bits-div">
					<label for="pa-atten-bits">Atten. Bits</label>
					<input type="number" min="0" max="10" value="0" step="1" id="pa-atten-bits" name="pa-atten-bits" />
				</div>
				<div class="form-group" id="pa-atten-lsb-div">
					<label for="pa-atten-lsb">Atten. LSB (dB)</label>
					<input type="number" min="0" max="5" value="0.5" step="0.25" id="pa-atten-lsb" name="pa-atten-lsb" />
				</div>
				<div class="form-group" style="font-size:0.7em;font-style: italic;">
					0 bits would be no quantization.
				</div>
			</div>
		</div>
		<div class="pa-update-div">
			<div style="display:flex; gap: 4px; justify-content: center;"><button id="pa-refresh">Update</button><button id="pa-reset">Reset</button></div>
			<progress id="pa-progress" max="100" value="70"></progress>
			<div id="pa-status">Loading...</div>
		</div>
	</div>
	<div class="canvas-grid">
		<div class="canvas-container">
			<div class="canvas-header"><h2>Element Phase</h2><span>&nbsp;</span></div>
			<div class="canvas-wrapper">
				<canvas id="pa-geometry-phase-canvas" class="canvas-grid"></canvas>
			</div>
			<div class="canvas-footer footer-group">
				<div>
					<label for="pa-geometry-phase-colormap">Colormap</label>
					<select id="pa-geometry-phase-colormap" name="pa-geometry-phase-colormap"></select>
				</div>
			</div>
		</div>
		<div class="canvas-container">
			<div class="canvas-header"><h2>Element Attenuation</h2><span>&nbsp;</span></div>
			<div class="canvas-wrapper">
				<canvas id="pa-geometry-magnitude-canvas" class="canvas-grid"></canvas>
			</div>
			<div class="canvas-footer footer-group">
				<div>
					<label for="pa-atten-scale">Scale</label>
					<input type="number" max="200" min="5" value="40" id="pa-atten-scale" name="pa-atten-scale">
				</div>
				<div>
					<label for="pa-geometry-magnitude-colormap">Colormap</label>
					<select id="pa-geometry-magnitude-colormap" name="pa-geometry-magnitude-colormap"></select>
				</div>
			</div>
		</div>
		<div class="canvas-container">
			<div class="canvas-header"><h2>2-D Radiation Pattern</h2><span id="pa-directivity-max">&nbsp;</span></div>
			<div class="canvas-wrapper">
				<canvas id="pa-farfield-canvas-2d" class="canvas-grid"></canvas>
			</div>
			<div class="canvas-footer">
				<div class="footer-group">
					<div>
						<label for="pa-farfield-domain">Domain</label>
						<select id="pa-farfield-domain"></select>
					</div>
					<div>
						<label for="pa-farfield-2d-scale">Scale</label>
						<input type="number" max="200" min="5" value="40" id="pa-farfield-2d-scale" name="pa-farfield-2d-scale">
					</div>
					<div>
						<label for="pa-farfield-2d-colormap">Colormap</label>
						<select id="pa-farfield-2d-colormap" name="pa-farfield-2d-colormap"></select>
					</div>
				</div>
				<div class="footer-group">
					<div>
						<label for="pa-farfield-ax1-points">Theta Points</label>
						<input type="number" min="11" max="513" value="257" id="pa-farfield-ax1-points" name="pa-farfield-ax1-points">
					</div>
					<div>
						<label for="pa-farfield-ax2-points">Phi Points</label>
						<input type="number" min="11" max="513" value="257" id="pa-farfield-ax2-points" name="pa-farfield-ax2-points">
					</div>
				</div>
			</div>
		</div>
	</div>
	<div class="canvas-full">
		<div class="canvas-container">
			<div class="canvas-header"><h2>1-D Pattern Cuts</h2></div>
			<div class="canvas-wrapper">
				<canvas id="pa-farfield-canvas-1d"></canvas>
			</div>
			<div class="canvas-footer">
				<div class="canvas-legend">
					<span class="legend-item" data-phi="0" data-v="0.0" data-az="0.0" data-visible="true">Phi = 0 deg</span>
					<span class="legend-item" data-phi="90" data-u="0.0" data-el="0.0" data-visible="true">Phi = 90 deg</span>
					<span style='font-size:0.8em'>Click to hide/show trace.</span>
				</div>
				<div>
					<label for="pa-farfield-1d-scale">Scale</label>
					<input type="number" max="200" min="5" value="40" id="pa-farfield-1d-scale" name="pa-farfield-1d-scale">
					<label for="pa-farfield-1d-colormap">Colormap</label>
					<select id="pa-farfield-1d-colormap" name="pa-farfield-1d-colormap"></select>
				</div>
			</div>
		</div>
	</div>
	<div class="canvas-full">
		<div class="canvas-container">
			<div class="canvas-header"><h2>Taper</h2></div>
			<div class="canvas-wrapper">
				<canvas id="pa-taper-canvas-1d"></canvas>
			</div>
			<div class="canvas-footer">
				<div class="canvas-legend">
					<span class="legend-item" data-axis="x" data-visible="true">X-Axis</span>
					<span class="legend-item" data-axis="y" data-visible="true">Y-Axis</span>
					<span style='font-size:0.8em'>Click to hide/show trace.</span>
				</div>
				<div>
					<label for="pa-taper-1d-colormap">Colormap</label>
					<select id="pa-taper-1d-colormap" name="pa-taper-1d-colormap"></select>
				</div>
			</div>
		</div>
	</div>



afterafterafterafterafterafterafterafterafterafterafterafter afterafterafterafter afterafterafterafterafterafterafter

