import {SceneControl, SceneControlWithSelector, SceneControlWithSelectorAutoBuild, SceneParent} from "./scene/scene-abc.js";
import {FindSceneURL} from "./scene/scene-util.js";
import {ScenePlot1D} from "./scene/plot-1d/scene-plot-1d.js";
import {Geometries} from "./phasedarray/geometry.js";
import {PhasedArray} from "./phasedarray/phasedarray.js";
import {FarfieldDomains} from "./phasedarray/farfield.js"
import {SteeringDomains} from "./phasedarray/steering.js"
import {Tapers} from "./phasedarray/tapers.js"
import {linspace} from "./util.js";
/** @import { SceneQueue } from "./scene/scene-queue.js" */

export class SceneControlGeometry extends SceneControlWithSelectorAutoBuild{
	static autoUpdateURL = false;
	constructor(parent){
		super(parent, 'geometry', Geometries, parent.find_element('geometry-controls'));
		this.activeGeometry = null;
	}
	control_changed(key){
		super.control_changed(key);
		this.activeGeometry = null;
	}
	get calculationWaiting(){ return this.activeGeometry === null; }
	add_to_queue(queue){
		if (this.calculationWaiting){
			queue.add('Building geometry...', () => {
					this.activeGeometry = this.build_active_object();
					this.activeGeometry.build();
				}
			)
		}
	}
}

export class SceneControlPhasedArray extends SceneControl{
	static autoUpdateURL = false;
	constructor(parent){
		super(parent, ['phase-bits', 'atten-lsb', 'atten-bits', 'atten-manual', 'phase-manual']);
		this.pa = null;
		this.geometryControl = new SceneControlGeometry(this);
		this.taperControl = new SceneControlAllTapers(this);
		this.steerControl = new SceneControlSteeringDomain(this);
		this.add_event_types(
			'phased-array-changed',
			'phased-array-phase-changed',
			'phased-array-attenuation-changed',
		);
	}
	/**
	* Add callable objects to queue.
	*
	* @param {SceneQueue} queue
	*
	* @return {null}
	* */
	add_to_queue(queue){
		let needsPhase = this.steerControl.calculationWaiting;
		let needsAtten = this.taperControl.calculationWaiting;
		let needsRecalc = false;
		let needsPhaseQ = this.changed['phase-bits'];
		let needsAttenQ = this.changed['atten-bits'] || this.changed['atten-lsb'];
		this.farfieldNeedsCalculation = false
		this.geometryControl.add_to_queue(queue);
		this.taperControl.add_to_queue(queue);

		if (this.geometryControl.calculationWaiting || this.pa === null){
			queue.add('Updating array...', () => {
					let first = this.pa === null;
					this.pa = new PhasedArray(this.geometryControl.activeGeometry);
					this.trigger_event('phased-array-changed', this.pa);
					if (first) this.load_hidden_controls();
				}
			)
			needsPhase = true;
			needsAtten = true;
			this.farfieldNeedsCalculation = true;
		}
		if (this.pa !== null){
			needsRecalc = needsRecalc || this.pa.requestUpdate;
		}
		if (needsPhase){
			queue.add('Calculating phase...', () => {
				const [theta, phi] = this.steerControl.get_theta_phi();
				this.pa.set_theta_phi(theta, phi);
				this.pa.compute_phase();
			});
			needsRecalc = true;
		}
		if (needsAtten){
			this.taperControl.add_calculator_queue(queue, this);
			needsRecalc = true;
		}
		if (needsRecalc){
			queue.add('Calculating vector...', () => {
				this.pa.calculate_final_vector();
				this.update_hidden_controls();
			});
			needsPhaseQ = true;
			needsAttenQ = true;
		}
		if (needsPhaseQ){
			queue.add('Quantizing phase...', () => {
				const bits = Math.max(0, Math.min(10, this.find_element('phase-bits').value));
				this.pa.quantize_phase(bits);
				this.trigger_event('phased-array-phase-changed', this.pa);
				this.clear_changed('phase-bits');
			});
			this.farfieldNeedsCalculation = true;
		}
		if (needsAttenQ){
			queue.add('Quantizing attenuation...', () => {
				const bits = Math.max(0, Math.min(10, this.find_element('atten-bits').value));
				const lsb = Math.max(0, Math.min(5, this.find_element('atten-lsb').value));
				this.pa.quantize_attenuation(bits, lsb);
				this.trigger_event('phased-array-attenuation-changed', this.pa);
				this.clear_changed('atten-bits', 'atten-lsb');
			});
			this.farfieldNeedsCalculation = true;
		}
		this.phaseChanged = needsPhase;
		this.attenChanged = needsAtten;
	}
	update_hidden_controls(){
		const mconfig = {};
		const pconfig = {};
		const mele = this.find_element('atten-manual');
		const pele = this.find_element('phase-manual');
		const pa = this.pa;
		for (let i = 0; i < pa.size; i++){
			if (!pa.vectorMagIsManual[i]) continue;
			mconfig[i] = [pa.vectorMagManual[i], pa.elementDisabled[i]];
		}
		if (Object.keys(mconfig).length === 0) mele.value = "";
		else mele.value = JSON.stringify(mconfig);
		mele.dispatchEvent(new Event('change'))
		for (let i = 0; i < pa.size; i++){
			if (!pa.vectorPhaseIsManual[i]) continue;
			pconfig[i] = pa.vectorPhaseManual[i];
		}
		if (Object.keys(pconfig).length === 0) pele.value = "";
		else pele.value = JSON.stringify(pconfig);
		pele.dispatchEvent(new Event('change'))
		const url = FindSceneURL();
		url.check_element('atten-manual', mele);
		url.check_element('phase-manual', pele);
	}
	load_hidden_controls(){
		const pa = this.pa;
		const mele = this.find_element('atten-manual');
		const pele = this.find_element('phase-manual');
		try{
			if (mele.value != ""){
				const mconfig = JSON.parse(mele.value);
				for (let i = 0; i < pa.size; i++){
					if (mconfig[i] === undefined) continue;
					const [v, d] = mconfig[i];
					pa.set_manual_magnitude(i, true, v, d);
				}
			}
		}
		catch(error){ console.log(error); }
		try{
			if (pele.value != ""){
				const pconfig = JSON.parse(pele.value);
				for (let i = 0; i < pa.size; i++){
					if (pconfig[i] === undefined) continue;
					pa.set_manual_phase(i, true, pconfig[i]);
				}
			}
		}
		catch(error){ console.log(error); }
	}
}

export class SceneControlTaper extends SceneControlWithSelectorAutoBuild{
	static autoUpdateURL = false;
	constructor(parent, key, htmlElement){
		super(parent, 'taper', Tapers, htmlElement, key);
		this.activeTaper = null;
	}
	control_changed(key){
		super.control_changed(key);
		this.activeTaper = null;
	}
	get calculationWaiting(){
		return this.activeTaper === null;
	}
	/**
	* Add callable objects to queue.
	*
	* @param {SceneQueue} queue
	*
	* @return {null}
	* */
	add_to_queue(queue){
		if (this.calculationWaiting){
			queue.add('Building Taper...', () => {
					this.activeTaper = this.build_active_object();
				}
			)
		}
	}
	/**
	* Build a taper control object.
	*
	* @param {SceneParent} parent
	* @param {String} key "x" or "y"
	*
	* @return {SceneControlTaper}
	* */
	static build(parent, key){
		const element = parent.find_element('taper-' + key + '-group')
		const k = parent.prepend + "-" + key + "-taper";
		const _create_group = (p) => {
			let kk = k;
			if (p !== undefined) kk += "-" + p;
			kk += "-div";
			var div = document.createElement('div');
			div.className = 'form-group';
			div.id = kk;
			element.appendChild(div);
			return div;
		}
		const _create_lbl = (div, p) => {
			let kk = k;
			if (p !== undefined) kk += "-" + p;
			const lbl = document.createElement("label");
			lbl.setAttribute("for", kk);
			div.appendChild(lbl);
			return lbl;
		}
		const _create_input = (div, p) => {
			let kk = k;
			if (p !== undefined) kk += "-" + p;
			const inp = document.createElement("input");
			inp.id = kk;
			inp.setAttribute('type', 'Number');
			inp.setAttribute('min', "0");
			inp.setAttribute('max', "100");
			inp.setAttribute('name', kk);
			inp.setAttribute('value', "0");
			div.appendChild(inp);
			return inp;
		}

		const div0 = _create_group();
		const div1 = _create_group('par-1');
		const div2 = _create_group('par-2');

		const lbl0 = _create_lbl(div0);
		lbl0.innerHTML = key.toUpperCase() + "-Taper";

		const sel0 = document.createElement("select");
		sel0.id = k;
		div0.appendChild(sel0);

		_create_lbl(div1, 'par-1');
		_create_input(div1, 'par-1');
		_create_lbl(div2, 'par-2');
		_create_input(div2, 'par-2');
		return new SceneControlTaper(parent, key);
	}
}

export class SceneControlAllTapers extends SceneControl{
	static autoUpdateURL = false;
	constructor(parent){
		super(parent, ['taper-sampling']);
		this.xControl = new SceneControlTaper(parent, 'x', parent.find_element('taper-x-group'));
		this.yControl = new SceneControlTaper(parent, 'y', parent.find_element('taper-y-group'));
		this.add_event_types('taper-changed');
	}
	get calculationWaiting(){
		return (
			this.xControl.calculationWaiting
			|| this.yControl.calculationWaiting
			|| this.changed['taper-sampling']
		);
	}
	control_changed(key){
		super.control_changed(key);
		const eleX = this.parent.find_element('taper-x-group');
		const eleY = this.parent.find_element('taper-y-group');
		if (this.find_element('taper-sampling')[1].selected){
			eleY.style.display = 'none';
			eleX.querySelector("label").innerHTML = "R-Taper";
		}
		else{
			eleY.style.display = 'block';
			eleX.querySelector("label").innerHTML = "X-Taper";
		}
	}
	/**
	* Add callable objects to queue.
	*
	* @param {SceneQueue} queue
	*
	* @return {null}
	* */
	add_to_queue(queue){
		this.xControl.add_to_queue(queue);
		this.yControl.add_to_queue(queue);
	}
	/**
	* Add callable objects to queue AFTER phased array
	* is created.
	*
	* @param {SceneQueue} queue
	* @param {SceneControlPhasedArray} src
	*
	* @return {null}
	* */
	add_calculator_queue(queue, src){
		if (this.find_element('taper-sampling')[0].selected){
			let taperX, taperY;
			// we're doing x/y sampling.
			queue.add("Calculating X taper...", () => {
				this.clear_changed('taper-sampling');
				const t = this.xControl.activeTaper;
				const geo = src.pa.geometry;
				taperX = t.calculate_from_geometry(geo.x, geo.dx);
			});
			queue.add("Calculating Y taper...", () => {
				const t = this.yControl.activeTaper;
				const geo = src.pa.geometry;
				taperY = t.calculate_from_geometry(geo.y, geo.dy);
			});
			queue.add("Multiplying tapers...", () => {
				this.trigger_event('taper-changed');
				src.pa.set_magnitude_weight(Float32Array.from(taperX, (x, i) => x * taperY[i]));
			});
		}
		else{
			// we're doing r sampling.
			queue.add("Calculating taper...", () => {
				const t = this.xControl.activeTaper;
				const geo = src.pa.geometry;
				src.pa.set_magnitude_weight(t.calculate_from_radial_geometry(geo.x, geo.y, geo.dx, geo.dy));
				this.trigger_event('taper-changed');
			});
		}
	}
	create_samples(points, axis){
		const x = linspace(-1, 1, points);
		const dx = x[1] - x[0];
		let y;
		if (this.find_element('taper-sampling')[0].selected){
			if (axis == 'x') y = this.xControl.activeTaper.calculate_from_geometry(x, dx);
			else y = this.yControl.activeTaper.calculate_from_geometry(x, dx);
		}
		else y = this.xControl.activeTaper.calculate_from_geometry(x, dx);
		return [x, y];
	}
}

export class SceneControlFarfieldDomain extends SceneControlWithSelector{
	static autoUpdateURL = false;
	constructor(parent, key){
		super(parent, key, FarfieldDomains);
		this.ff = null;
		this.validMaxMonitors = new Set(['directivity']);
		this.maxMonitors = {};
		this.add_event_types('farfield-changed', 'farfield-calculation-complete');
	}
	/**
	* Add callable functions to monitor values.
	*
	* @param {string} key Examples: directivity
	* @param {function(Number):null} callback
	*
	* @return {null}
	* */
	add_max_monitor(key, callback){
		if (!(this.validMaxMonitors.has(key))){
			throw Error(`Invalid monitor ${key}. Expected: ${Array.from(this.validMaxMonitors).join(', ')}`)
		}
		if (!(key in this.maxMonitors)) this.maxMonitors[key] = [];
		this.maxMonitors[key].push(callback);
	}
	/**
	* Add callable objects to queue.
	*
	* @param {SceneQueue} queue
	*
	* @return {null}
	* */
	add_to_queue(queue){
		const arrayControl = this.parent.arrayControl;
		let needsRecalc = arrayControl.farfieldNeedsCalculation;

		if (this.changed['farfield-ax1-points'] || this.changed['farfield-ax2-points'] || this.ff === null){
			queue.add('Creating farfield mesh...', () => {
				this.ff = this.build_active_object();
				this.trigger_event('farfield-changed', this.ff);
			});
			needsRecalc = true;
		}
		if (needsRecalc){
			queue.add_iterator('Calculating farfield...', () => {
				return this.ff.calculator_loop(arrayControl.pa)
			});
			queue.add("Notifying farfield change...", () => {
				this.trigger_event('farfield-calculation-complete', this.ff);
				for (const [key, value] of Object.entries(this.maxMonitors)){
					let val;
					if (key == 'directivity') val = this.ff.dirMax;
					else throw Error(`Unknown max key ${key}.`)
					value.forEach((e) => e(val));
				}
			})
		}
		this.needsRedraw = needsRecalc;
	}
}

export class SceneControlSteeringDomain extends SceneControlWithSelector{
	static autoUpdateURL = false;
	constructor(parent){
		super(parent, 'steering-domain', SteeringDomains);
		this._last = this.selected_class();
	}
	get calculationWaiting(){return this.changed['theta'] || this.changed['phi'] || this.changed['steering-domain']};
	control_changed(key){
		if (key == this.primaryKey){
			if (this._last === undefined) return;
			const c1 = this.find_object_map('theta');
			const c2 = this.find_object_map('phi');
			const p1 = Number(c1.ele.value);
			const p2 = Number(c2.ele.value);
			const obj = this.build_active_object();
			let [n1, n2] = obj.from(this._last.title, p1, p2);
			if (isNaN(n1) || isNaN(n2)){
				n1 = 0.0;
				n2 = 0.0;
			}
			c1.set_value(n1);
			c2.set_value(n2);
			this._last = this.selected_class();
		}
		super.control_changed(key);
	}
	get_theta_phi(){
		const obj = this.build_active_object();
		this.clear_changed('theta', 'phi', 'steering-domain');
		return [obj.theta_deg, obj.phi_deg];
	}
}

export class SceneTaperCuts extends ScenePlot1D{
	draw(){
		this.reset();
		this.set_xlabel('Window');
		this.set_ylabel('Magnitude');
		this.set_xgrid(-0.5, 0.5, 11);
		this.set_xgrid_points(1);

		const pa = this.arrayScene;
		if (pa === undefined || pa == null) return;
		const taper = pa.taperControl;
		if (taper === undefined || taper === null) return;

		let belowZero = false;
		this.legend_items().forEach((e) => {
			const v = e.getAttribute('data-axis');
			if (v !== null){
				const [x, y] = taper.create_samples(101, v);
				const maxV = Math.max(...Float32Array.from(y, (i) => Math.abs(i)));
				const minV = Math.min(...y);
				if (minV < 0) belowZero = true;
				if (x !== null) this.add_data(x, Float32Array.from(y, (i) => i/maxV), e);
			}
		});
		if (belowZero) this.set_ygrid(-1, 1, 11);
		else this.set_ygrid(0, 1, 11);
		super.draw();
	}
	/**
	* Bind a Phased Array Scene.
	*
	* @param {SceneControlPhasedArray} scene
	*
	* @return {null}
	* */
	bind_phased_array_scene(scene){
		this.arrayScene = scene;
		scene.taperControl.addEventListener('taper-changed', () => {
			this.draw();
		});
	}
}
