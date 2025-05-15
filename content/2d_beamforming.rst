.. _2d-beamforming-chapter:

##############
2D Beamforming
##############

before

.. raw:: html

	<div class="text-group" style="text-align:center;">
		<h1>Phased Array Visualizer</h1>
		<p>By: <a href="https://github.com/jasondurbin" target="_blank">Jason Durbin</a> (<a href="https://www.linkedin.com/in/jasondurbin/" target="_blank">LinkedIn</a>).</p>
		<div>Dark Mode: <button class="theme-toggle">auto</button></div>
		<br>
	</div>
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
	<div class="body-content">
		<h2>About</h2>
		<p>This tool allows you to change a phased array's geometry, element spacing, steering position, add sidelobe tapering, and other features.</p>
		<p>This demo was created by <a href="https://github.com/jasondurbin" target="_blank">Jason Durbin</a>, a <a href="https://www.linkedin.com/in/jasondurbin/" target="_blank">free-lancing phased array engineer</a>.
		<h2>Usage and Notes</h2>
		<ul>
			<li>Antenna elements are assumed to be isotropic. However, the directivity calculation assumes half-hemisphere radiation (e.g. no back lobes). Therefore, the computed directivity will be 3 dBi higher than using pure isotropic. Said in different terms, the individual element gain is +3.0 dBi.</li>
			<li>The mesh can be made finer by increasing theta/phi, u/v, or az/el points. However, increasing the number of points may result in laggy performance.</li>
			<li>Clicking (or long pressing) elements in the phase/attenuation plots allows you to manually set phase/attenuation. <b>Be sure to select "enable override."</b> Additionally, the attenuation pop-up allows you to disable elements.</li>
			<li>Hovering (or touching) the 2-D farfield plot or geometry plots will show the value of the plot under the cursor.</li>
		</ul>
		<h2>Commercial Use</h2>
		<p>This tool or any derivatives of this tool may not be hosted on commercial websites (internal or external) without approval. Of course, you are welcome to share the URL.</p>
		<p>If your company sells phased arrays, beamforming ICs, or other phased array related products and you'd like host a bespoke version of this tool on your website, please contact <a href='mailto:hello@neonphysics.com'>hello@neonphysics.com</a></p>
		<p>Commercial use is otherwise prohibited.</p>
		<h2>Donation and Feedback</h2>
		<p>If you enjoy this visualizer, please consider donating <a href='https://www.paypal.com/donate/?business=D7S3JKRAAKUNQ&no_recurring=0&currency_code=USD' target="_blank">using PayPal</a>.</p>
		<p>If you have any recommendations, feedback, or requests, feel free to send Jason a message <a href='https://www.linkedin.com/in/jasondurbin/' target='_blank'>on LinkedIn</a> or send an email to <a href='mailto:hello@neonphysics.com'>hello@neonphysics.com</a>.
		<p><a href='https://www.paypal.com/donate/?business=D7S3JKRAAKUNQ&no_recurring=0&currency_code=USD' target="_blank"><img src='https://img.shields.io/badge/PayPal-Donate-fa448c?logo=paypal' alt='Donate through PayPal'></a></p>
		<h2>Attributions</h2>
		<p><ul>
			<li>Meshing colormaps are generated to match <a href='https://matplotlib.org/stable/users/explain/colors/colormaps.html' target="_blank">matplotlib's</a>.</li>
			<li>Listed colormaps are from <a href="https://sronpersonalpages.nl/~pault/" target="_blank">Paul Tol's Color Scheme</a>.</li>
			<li>Most of the tapers were pulled from <a href='https://www.researchgate.net/publication/316281181_Catalog_of_Window_Taper_Functions_for_Sidelobe_Control' target="_blank">"Catalog of Window Taper Functions for Sidelobe Control" by Armin W. Doerry</a>.</li>
		</ul></p>
		<h2>Tracking and Proprietary Information</h2>
		<p>I use a self-hosted version of <a href="https://matomo.org/" target="_blank">Matamo</a> to track viewers and analytics. I do <strong>not</strong> sell any information and I only use the information for my personal understanding of usage to better improve the tool.</p>
		<p>Please note: because the settings are saved in the URL, I can personally view and replicate any configuration. If you or your company deem any configuration to be proprietary, I do not recommend sharing the link. However, I have intentionally not included things like frequency to avoid this concern.</p>
	</div>
    <script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>


after
