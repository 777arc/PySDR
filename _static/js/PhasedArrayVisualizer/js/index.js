import {SceneControlPhasedArray, SceneControlFarfieldDomain, SceneTaperCuts} from "./index-scenes.js";
import {ScenePlotFarfieldCuts} from "./scene/plot-1d/scene-plot-farfield-cuts.js";
import {ScenePlotFarfield2D} from "./scene/plot-2d/scene-plot-2d-farfield.js";
import {ScenePlot2DGeometryPhase, ScenePlot2DGeometryAtten} from "./scene/plot-2d/scene-plot-2d-geometry.js";
import {SceneParent} from "./scene/scene-abc.js"
import {SceneTheme} from "./scene/scene-util.js";

document.addEventListener('DOMContentLoaded', () => {
	new SceneTheme();
	const scene = new PhasedArrayScene('pa');
	scene.build_queue();
});

/**	 *
 * Create scene for Phased Array simulator.
 *
 * @param {string} prepend - Prepend used on HTML IDs.
 * */
export class PhasedArrayScene extends SceneParent{
	constructor(prepend){
		super(prepend, ['refresh', 'reset'])
		this.create_queue(this.find_element('progress'), this.find_element('status'));
		this.arrayControl = new SceneControlPhasedArray(this);
		this.farfieldControl = new SceneControlFarfieldDomain(this, 'farfield-domain');

		this.plotFF = new ScenePlotFarfield2D(this, this.find_element('farfield-canvas-2d'), 'farfield-2d-colormap');
		this.plot1D = new ScenePlotFarfieldCuts(this, this.find_element('farfield-canvas-1d'), 'farfield-1d-colormap');
		this.plotTaper = new SceneTaperCuts(this, this.find_element('taper-canvas-1d'), 'taper-1d-colormap');
		this.geoPhase = new ScenePlot2DGeometryPhase(this, this.find_element('geometry-phase-canvas'), 'geometry-phase-colormap');
		this.geoAtten = new ScenePlot2DGeometryAtten(this, this.find_element('geometry-magnitude-canvas'), 'geometry-magnitude-colormap');
		this.geoAtten.install_scale_control('atten-scale');

		this.geoPhase.bind_phased_array_scene(this.arrayControl);
		this.geoAtten.bind_phased_array_scene(this.arrayControl);
		this.plot1D.bind_farfield_scene(this.farfieldControl);
		this.plotFF.bind_farfield_scene(this.farfieldControl);
		this.plotTaper.bind_phased_array_scene(this.arrayControl);
		this.plot1D.install_scale_control('farfield-1d-scale');
		this.plotFF.install_scale_control('farfield-2d-scale');
		this.farfieldControl.add_max_monitor('directivity', (v) => {
			this.find_element('directivity-max').innerHTML = `Directivity: ${(10*Math.log10(v)).toFixed(2)} dB`
		});

		this.find_element('refresh').addEventListener('click', () => {
			this.update_url_parameters();
			this.build_queue();
		});
		this.find_element('reset').addEventListener('click', () => {
			this.reset_url_parameters();
			this.build_queue();
		});
		this.create_popup_overlay();
		this.bind_url_elements();
	}
	build_queue(){
		this.queue.reset();
		this.arrayControl.add_to_queue(this.queue);
		this.farfieldControl.add_to_queue(this.queue);
		this.queue.start();
	}
}
