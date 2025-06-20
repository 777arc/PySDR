import {rad2deg} from "../../util.js";
import {ScenePlot1D} from "./scene-plot-1d.js";
/** @import { FarfieldHint } from "../../phasedarray/farfield.js" */
/** @import { SceneControlFarfieldDomain } from "../../index-scenes.js" */

export class ScenePlotFarfieldCuts extends ScenePlot1D{
	constructor(parent, canvas, cmapKey){
		super(parent, canvas, cmapKey);
		this.add_event_types('data-min-changed');
		this.addEventListener('data-min-changed',() => {
			this.draw();
		})
		this.min = -40;
		this.engine = new FarfieldCutEngineSpherical(this);
	}
	/**
	* Load farfield object.
	*
	* @param {FarfieldHint} ff
	*
	* @return {null}
	* */
	load_farfield(ff){
		this.ff = ff;
		if (ff.domain == 'uv') this.engine = new FarfieldCutEngineUV(this);
		else if (ff.domain == 'ludwig3') this.engine = new FarfieldCutEngineLudwig3(this);
		else this.engine = new FarfieldCutEngineSpherical(this);
	}
	/**
	* Bind a Farfield Scene.
	*
	* @param {SceneControlFarfieldDomain} scene
	*
	* @return {null}
	* */
	bind_farfield_scene(scene){
		scene.addEventListener('farfield-calculation-complete', (ff) => {
			this.load_farfield(ff);
			this.draw();
		});
	}
	draw(){
		this.engine.draw();
		super.draw();
	}
}

export class FarfieldCutEngineABC{
	/**
	* Create 1D plotting engine.
	*
	* @param {ScenePlotFarfieldCuts} parent
	* */
	constructor(parent){
		this.parent = parent;
	}
	legend_items(){ return this.parent.legend_items(); }
	get ff(){ return this.parent.ff; }
}

export class FarfieldCutEngineSpherical extends FarfieldCutEngineABC{
	create_grid(){
		this.parent.reset();
		this.parent.set_xlabel('Theta (deg)');
		this.parent.set_ylabel('Relative Directivity (dB)');
		this.parent.set_xgrid(-90, 90, 13);
		this.parent.set_ygrid(this.parent.min, 0, 11);
	}
	draw(){
		const ff = this.ff;
		this.create_grid();
		if (ff === undefined || ff == null) return;
		this.legend_items().forEach((e) => {
			let v = e.getAttribute('data-phi');
			if (v !== null){
				let [x, y] = ff.constant_phi(v);
				e.innerHTML = `phi = ${v} deg`
				if (x !== null) this.parent.add_data(rad2deg(x), y, e);
			}
		});
	}
}

export class FarfieldCutEngineUV extends FarfieldCutEngineABC{
	create_grid(){
		this.parent.reset();
		this.parent.set_xlabel('u/v');
		this.parent.set_ylabel('Relative Directivity (dB)');
		let xmin = -1, xmax = 1;
		const ff = this.ff;
		if (ff !== undefined){
			xmin = ff.u[0]
			xmax = ff.u[ff.u.length - 1]
		}
		this.parent.set_xgrid(xmin, xmax, 11);
		this.parent.set_ygrid(this.parent.min, 0, 11);
		this.parent.set_xgrid_points(1);
	}
	draw(){
		const ff = this.ff;
		this.create_grid();
		if (ff === undefined || ff == null) return;
		this.legend_items().forEach((e) => {
			const iu = e.getAttribute('data-u');
			if (iu !== null){
				let [x, y] = ff.constant_u(iu);
				e.innerHTML = `u = ${iu}`
				if (x !== null) this.parent.add_data(x, y, e);
			}
			const iv = e.getAttribute('data-v');
			if (iv !== null){
				let [x, y] = ff.constant_v(iv);
				e.innerHTML = `v = ${iv}`
				if (x !== null) this.parent.add_data(x, y, e);
			}
		});
	}
}

export class FarfieldCutEngineLudwig3 extends FarfieldCutEngineABC{
	create_grid(){
		this.parent.reset();
		this.parent.set_xlabel('Az/El (deg)');
		this.parent.set_ylabel('Relative Directivity (dB)');
		this.parent.set_xgrid(-90, 90, 13);
		this.parent.set_ygrid(this.parent.min, 0, 11);
		this.parent.set_xgrid_points(0);
		this.parent.set_ygrid_points(0);
	}
	draw(){
		const ff = this.ff;
		this.create_grid();
		if (ff === undefined || ff == null) return;
		this.legend_items().forEach((e) => {
			let v = e.getAttribute('data-az');
			if (v !== null){
				let [x, y] = ff.constant_az(v);
				e.innerHTML = `Az = ${v} deg`
				if (x !== null) this.parent.add_data(rad2deg(x), y, e);
			}
			v = e.getAttribute('data-el');
			if (v !== null){
				let [x, y] = ff.constant_el(v);
				e.innerHTML = `El = ${v} deg`
				if (x !== null) this.parent.add_data(rad2deg(x), y, e);
			}
		});
	}
}
