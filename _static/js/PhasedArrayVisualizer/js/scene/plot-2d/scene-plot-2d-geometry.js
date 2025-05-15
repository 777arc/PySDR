import {ScenePlotABC} from "../scene-plot-abc.js"
/** @import { SceneControlPhasedArray } from "../../index-scenes.js"*/
/** @import { PhasedArray } from "../../phasedarray/phasedarray.js"*/

export class ScenePlot2DGeometryABC extends ScenePlotABC{
	constructor(parent, canvas, cmapKey, defaultCMAP, strokeColor){
		if (defaultCMAP === undefined) defaultCMAP = 'viridis';
		if (strokeColor === undefined) strokeColor = 'black'
		let cmap = parent.create_mesh_colormap_selector(cmapKey, defaultCMAP);
		super(parent, canvas, cmap);
		this.strokeColor = strokeColor;
		this.cmap.addEventListener('change', () => {this.build_queue();})
		this.create_hover_items();
		this.create_progress_bar();
		this.create_queue();
	}
	event_to_id(e){
		let i = null;
		if (e.isTrusted && this.pa !== undefined){
			const f = this.canvas.index_from_event;
			if (f !== undefined) i = f(e);
		}
		return i;
	}
	install_hover_item(callback){
		this.canvas.addEventListener('mousemove', (e) => {
			if (this.queue.running) return;
			let i = this.event_to_id(e);
			let text = "&nbsp;";
			if (i !== null){
				const geo = this.pa.geometry;
				const t = callback(i);
				text = `Element[${i}] (${geo.x[i].toFixed(2)}, ${geo.y[i].toFixed(2)}): ${t}`
			}
			this.canvas.hover_container.innerHTML = text;
		});
	}
	get isValid(){ return (this.pa !== undefined && this.pa !== null); }
	/**
	* Load Phased array object.
	*
	* @param {PhasedArray} pa
	*
	* @return {null}
	* */
	load_phased_array(pa){ this.pa = pa; }
	/**
	* Bind a Phased Array Scene.
	*
	* @param {SceneControlPhasedArray} scene
	*
	* @return {null}
	* */
	bind_phased_array_scene(scene){
		scene.addEventListener('phased-array-changed', (pa) => this.load_phased_array(pa));
	}
	draw(data){
		if (this.pa === undefined) return;
		const canvas = this.canvas;
		const colormap = this.cmap.cmap();
		const scale = 600;
		const geo = this.pa.geometry;
		canvas.width = scale;
		canvas.height = scale;
		const ctx = canvas.getContext('2d');
		this.cmap.changed = false;
		ctx.reset();

		const maxX = Math.max(...geo.x) + geo.dx/2;
		const minX = Math.min(...geo.x) - geo.dx/2;
		const maxY = Math.max(...geo.y) + geo.dy/2;
		const minY = Math.min(...geo.y) - geo.dy/2;

		const wx = (maxX - minX);
		const wy = (maxY - minY);
		const sc = Math.min(canvas.width/wx, canvas.height/wy);
		const ox = (canvas.width - wx*sc)/2 - minX*sc;
		const oy = (canvas.height - wy*sc)/2 - minY*sc;
		const dx = geo.dx*0.98*sc/2;
		const dy = geo.dy*0.98*sc/2;

		const _xy_to_wh = (x, y) => [x*sc+ox, scale-(y*sc+oy)];
		const _wh_to_xy = (w, h) => [(w-ox)/sc, (scale-h-oy)/sc];

		canvas.transform_to_xy = _wh_to_xy;
		canvas.transform_to_wh = _xy_to_wh;
		canvas.index_from_event = (e) => {
			const rect = canvas.getBoundingClientRect();
			let ex, ey;
			if (e.type == 'touchstart'){
				ex = e.touches[0].clientX;
				ey = e.touches[0].clientY;
			}
			else{
				ex = e.clientX;
				ey = e.clientY;
			}
			const wx = (ex - rect.left)/rect.width*canvas.width;
			const wy = (ey - rect.top)/rect.height*canvas.height;
			const [x, y] = _wh_to_xy(wx, wy);
			const dx = geo.dx/2;
			const dy = geo.dy/2;
			let eleI = null;
			for (let i = 0; i < geo.length; i++){
				if (((x - geo.x[i])/dx)**2 + ((y - geo.y[i])/dy)**2 <= 1) {
					eleI = i;
					break;
				}
			}
			return eleI;
		};
		for (let i = 0; i < geo.length; i++){
			let [x, y] = _xy_to_wh(geo.x[i], geo.y[i]);
			ctx.beginPath();
			ctx.ellipse(x, y, dx, dy, 0.0, 0.0, 2*Math.PI);
			ctx.closePath();
			ctx.fillStyle = colormap(data[i]);
			ctx.fill();
			ctx.strokeStyle = this.strokeColor;
			ctx.stroke();
		}
	}
	build_queue(){ throw Error("Don't call generic build_queue."); }
	install_popup(dtype, controls, changedCallback, updaterCallback, clearAllCallback){
		const _show_popup = (e) => {
			if (e.pointerType == 'touch' && e.type == 'click') return;
			if (this.queue.running) return;
			if (!this.isValid) return;
			let i = this.event_to_id(e);
			if (i === null) return;
			const pa = this.pa;

			const dcontrols = [{
				'label': "Element ID",
				'type': 'number',
				'min': 0,
				'max': pa.geometry.length - 1,
				'id': 'index',
				'value': i,
			},{
				'label': "Location:",
				'type': 'span',
				'id': 'loc',
			},{
				'label': `Current ${dtype}:`,
				'type': 'span',
				'id': 'current-value',
			},{
				'label': "Enable Override",
				'type': 'checkbox',
				'id': 'override',
				'value': true,
			}];
			controls.forEach((e) => dcontrols.push(e));
			const popup = this.create_popup("Manually Change " + dtype, dcontrols, (config) => {
				if (config === null) return;
				changedCallback(_i(), config);
				this.parent.build_queue();
			});
			const lbl = popup.element('loc');
			const _i = () => Math.max(0, Math.min(pa.geometry.length - 1, popup.element('index').value));
			const _update = () => {
				const i = _i();
				const res = updaterCallback(i);
				lbl.innerHTML = `(${pa.geometry.x[i].toFixed(2)}, ${pa.geometry.y[i].toFixed(2)})`;
				for (const [key, value] of Object.entries(res)) popup.set_element_value(key, value);
			}
			popup.element('index').addEventListener('change', _update);
			popup.add_action(`Clear All ${dtype} Overrides`).addEventListener('click', () => {
				clearAllCallback();
				this.parent.build_queue();
			});
			popup.add_note(
				'To override phase/attenuation, select "Enable Override" '
				+ 'and enter the desired value.', 'popup-note');
			_update();
			popup.show_from_event(e);
		}
		this.canvas.addEventListener('click', _show_popup);
		const onlongpress = (ele, cb) => {
			let tid;
			ele.addEventListener('touchstart', (e) => {
				tid = setTimeout(() => {
					tid = null;
					e.stopPropagation();
					cb(e);
				}, 500);
			});
			ele.addEventListener('contextmenu', (e) => { e.preventDefault(); });
			ele.addEventListener('touchend', (e) => { if (tid) clearTimeout(tid); else e.preventDefault();});
			ele.addEventListener('touchmove', () => { if (tid) clearTimeout(tid);});
		}
		onlongpress(this.canvas, _show_popup);
	}
}

export class ScenePlot2DGeometryPhase extends ScenePlot2DGeometryABC{
	constructor(parent, canvas, cmapKey, min, max){
		super(parent, canvas, cmapKey, 'hsv');
		if (min === undefined) min = -180;
		if (max === undefined) max = 180;
		this.min = min;
		this.max = max;
		this.install_hover_item((i) => `${(this.pa.vectorPhase[i]*180/Math.PI).toFixed(2)} deg`);
		this._needsRescale = false;
		this.install_popup('Phase', [{
			'label': `Manual Phase (deg)`,
			'type': 'number',
			'min': 0,
			'max': 360,
			'id': 'value',
			'value': 0,
			'focus': true,
		}], (i, config) => {
			this.pa.set_manual_phase(i, config['override'], config['value']*Math.PI/180);
		}, (i) => {
			let ov;
			if (this.pa.vectorPhaseIsManual[i]) ov = this.pa.vectorPhaseManual[i];
			else ov = this.pa.vectorPhase[i];
			const nv = (ov*180/Math.PI).toFixed(2);
			return {
				'value': nv,
				'override': this.pa.vectorPhaseIsManual[i],
				'current-value': `${nv} deg`
			}
		}, () => { this.pa.clear_all_manual_phase();})
	}
	/**
	* Bind a Phased Array Scene.
	*
	* @param {SceneControlPhasedArray} scene
	*
	* @return {null}
	* */
	bind_phased_array_scene(scene){
		super.bind_phased_array_scene(scene);
		scene.addEventListener('phased-array-phase-changed', () => {
			this._needsRescale = true;
			this.build_queue();
		});
	}
	build_queue(){
		this.queue.reset();
		if (this._needsRescale){
			this.queue.add('Rescaling phase...', () => {
				this.rescale_phase();
			});
		}
		this.queue.add('Drawing phase...', () => {
			this.draw();
		});
		this.queue.start("&nbsp;");
	}
	rescale_phase(){
		const pa = this.pa;
		const phaseMin = this.min;
		const phaseMax = this.max;
		const pd = phaseMax - phaseMin;
		this.vectorPhaseScale = new Float32Array(pa.geometry.length);
		for (let i = 0; i < pa.geometry.length; i++){
			let pha = pa.vectorPhase[i]*180/Math.PI;
			while (pha > 180) pha -= 360;
			while (pha < -180) pha += 360;
			this.vectorPhaseScale[i] = (pha - phaseMin)/pd;
		}
		this._needsRescale = false;
	}
	draw(){ return super.draw(this.vectorPhaseScale); }
}

export class ScenePlot2DGeometryAtten extends ScenePlot2DGeometryABC{
	constructor(parent, canvas, cmapKey, min, max){
		super(parent, canvas, cmapKey, 'inferno_r');
		this.add_event_types('data-min-changed');
		if (min === undefined) min = -40;
		if (max === undefined) max = 0;
		this.min = min;
		this.max = max;
		this.install_hover_item((i) => `${this.pa.vectorAtten[i].toFixed(2)} dB`);
		this.addEventListener('data-min-changed', () => {this.build_queue(true);})
		this._needsRescale = true;
		this.install_popup('Attenuation', [{
			'label': `Manual Attenuation (dB)`,
			'type': 'number',
			'min': -100,
			'max': 100,
			'id': 'value',
			'step': 'none',
			'value': 0,
			'focus': true,
		},{
			'label': `Disable Element`,
			'type': 'checkbox',
			'id': 'disabled',
		}], (i, config) => {
			this.pa.set_manual_magnitude(i, config['override'], 10**(-Math.abs(config['value'])/20), config['disabled']);
		}, (i) => {
			let ov;
			if (this.pa.vectorMagIsManual[i]) ov = this.pa.vectorMagManual[i];
			else ov = this.pa.vectorMag[i];
			const nv = (20*Math.log10(Math.abs(ov))).toFixed(2);
			return {
				'value': nv,
				'override': this.pa.vectorMagIsManual[i],
				'current-value': `${nv} dB`,
				'disabled': this.pa.elementDisabled[i],
			}
		}, () => { this.pa.clear_all_manual_magnitude();})
	}
	rescale_atten(){
		const pa = this.pa;
		const attenMin = this.min;
		const attenMax = this.max;
		const am = attenMax - attenMin;
		const ma = Math.max(...pa.vectorAtten);
		this.vectorAttenScaled = new Float32Array(pa.geometry.length);
		for (let i = 0; i < pa.geometry.length; i++){
			this.vectorAttenScaled[i] = -(pa.vectorAtten[i] - ma - attenMax)/am;
		}
		this._needsRescale = false;
	}
	draw(){ return super.draw(this.vectorAttenScaled); }
	/**
	* Bind a Phased Array Scene.
	*
	* @param {SceneControlPhasedArray} scene
	*
	* @return {null}
	* */
	bind_phased_array_scene(scene){
		super.bind_phased_array_scene(scene);
		scene.addEventListener('phased-array-attenuation-changed', () => {
			this._needsRescale = true;
			this.build_queue();
		});
	}
	build_queue(){
		this.queue.reset();
		if (this._needsRescale){
			this.queue.add('Rescaling attenuation...', () => {
				this.rescale_atten();
			});
		}
		this.queue.add('Drawing attenuation...', () => {
			this.draw();
		});
		this.queue.start("&nbsp;");
	}
}
