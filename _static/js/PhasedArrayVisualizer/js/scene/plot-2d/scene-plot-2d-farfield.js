import {adjust_theta_phi} from "../../util.js";
import {ScenePlotABC} from "../scene-plot-abc.js"
/** @import { FarfieldHint } from "../../phasedarray/farfield.js" */
/** @import { SceneControlFarfieldDomain } from "../../index-scenes.js" */

const CANVAS_SCALER = 7000;
export class ScenePlotFarfield2D extends ScenePlotABC{
	constructor(parent, canvas, cmapKey){
		let cmap = parent.create_mesh_colormap_selector(cmapKey, 'viridis');
		super(parent, canvas, cmap);
		this.add_event_types('data-min-changed');
		canvas.addEventListener('mousemove', (e) => {
			if (this.queue.running) return;
			this.show_farfield_hover(e);
		});
		this.cmap.addEventListener('change', () => {this.build_queue();})
		this.ff = undefined;
		this.engine = undefined;
		this._needsRescale = false
		this.addEventListener('data-min-changed',() => {
			this._needsRescale = true
			this.build_queue();
		})
		this.create_hover_items();
		this.create_progress_bar();
		this.create_queue();
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
		if (ff.domain == 'spherical') this.engine = new PlotFarfield2DEngineSpherical(this);
		else if (ff.domain == 'uv') this.engine = new PlotFarfield2DEngineUV(this);
		else if (ff.domain == 'ludwig3') this.engine = new PlotFarfield2DEngineLudwig3(this);
		else throw Error(`Unknown farfield domain ${ff.domain}.`)
		this._needsRescale = true;
	}
	get isValid(){ return !(this.ff === undefined || this.ff === null); }
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
			this.build_queue();
		});
	}
	rescale(){
		if (!this.isValid) return;
		const logMin = -this.min;
		const ff = this.ff
		const [p1, p2] = ff.meshPoints;
		this.farfield_log_scale = new Array(p2);
		for (let ip = 0; ip < p2; ip++){
			const pc = ff.farfield_log[ip];
			this.farfield_log_scale[ip] = Float32Array.from({length: p1}, (_, it) => ((pc[it] + logMin)/logMin));
		}
		this._needsRescale = false;
	}
	create_colormap(){
		if (!this.isValid) return;
		const ff = this.ff
		const cmap = this.cmap.cmap();
		const [p1, p2] = ff.meshPoints;
		this.colormap_vals = new Array(p2);
		for (let ip = 0; ip < p2; ip++){
			const arr = new Array(p1);
			this.colormap_vals[ip] = arr;
			for (let it = 0; it < p1; it++){
				arr[it] = cmap(this.farfield_log_scale[ip][it]);
			}
		}
		cmap.changed = false;
	}
	build_queue(){
		this.queue.reset();
		if (this._needsRescale){
			this.queue.add('Rescaling farfield...', () => {
				this.rescale();
			});
		}
		this.queue.add('Creating farfield colormap...', () => {
			this.create_colormap();
		});
		this.queue.add('Drawing 2D farfield...', () => {
			this.engine.draw(this.colormap_vals);
			delete this.colormap_vals;
		});
		this.queue.start("&nbsp;");
	}
	show_farfield_hover(e){
		const canvas = this.canvas;
		const f = canvas.index_from_event;
		const ff = this.ff
		let text = "&nbsp;";
		if (f !== undefined && ff !== null && ff.dirMax != null){
			const res =  f(e);
			if (res !== null){
				const [x, y, it, ip] = res;
				const ff1 = 10*Math.log10(ff.farfield_total[ip][it]/ff.maxValue);
				const ff2 = ff1 + 10*Math.log10(ff.dirMax);
				text = `(${x.toFixed(2)}, ${y.toFixed(2)}): ${ff2.toFixed(2)} dBi (${ff1.toFixed(2)} dB)`;
			}
		}
		canvas.hover_container.innerHTML = text;
	}
}

export class PlotFarfield2DEngineABC{

	/**
	* Create a new 2D plotting engine.
	*
	* @param {ScenePlotFarfield2D} parent
	*
	* @return {null}
	* */
	constructor(parent){
		this.parent = parent;
	}
	get canvas(){ return this.parent.canvas; }
	get ff(){ return this.parent.ff; }
	get isValid(){ return this.parent.isValid; }
}

export class PlotFarfield2DEngineSpherical extends PlotFarfield2DEngineABC{
	constructor(parent, thetaSteps, phiSteps){
		super(parent);
		if (thetaSteps === undefined) thetaSteps = 7;
		if (phiSteps === undefined) phiSteps = 13;
		this.phiSteps = phiSteps;
		this.thetaSteps = thetaSteps;
	}
	draw(cmap_vals){
		const canvas = this.canvas;
		const ctx = canvas.getContext('2d');
		const ff = this.ff;
		ctx.reset();
		canvas.width = CANVAS_SCALER;
		canvas.height = CANVAS_SCALER;
		const thetaStep = Math.PI/(ff.thetaPoints - 1);
		const phiStep = Math.PI/(ff.phiPoints - 1);
		const r = Math.min(canvas.width/2, canvas.height/2);
		const ts = Math.PI + thetaStep;
		const smoothing = Math.min(0.01, thetaStep*0.5);
		const pci = (ff.phiPoints - 1)/2;
		const tci = (ff.thetaPoints - 1)/2;
		ctx.translate(canvas.width/2, canvas.height/2);
		ctx.scale(1.0, -1.0);

		canvas.index_from_event = (e) => {
			const rect = canvas.getBoundingClientRect();
			const u = 2*(e.clientX - rect.left)/rect.width - 1.0;
			const v = 1-2*(e.clientY - rect.top)/rect.height;

			const r = Math.sqrt(u**2 + v**2);
			if (r > 1) return null;
			const [th, ph] = adjust_theta_phi(r*Math.PI/2, Math.atan2(v, u), false);
			let it = Math.round((Math.PI/2 + th)/thetaStep);
			let ip = Math.round((Math.PI/2 + ph)/phiStep);
			if (it >= ff.thetaPoints) it = ff.thetaPoints - 1;
			if (ip >= ff.phiPoints) ip = ff.phiPoints - 1;
			if (it < 0) it = 0;
			if (ip < 0) ip = 0;
			return [ff.theta[it]*180/Math.PI, ff.phi[ip]*180/Math.PI, it, ip];
		};

		for (let it = 0; it < ff.thetaPoints; it++) {
			const r1 = Math.abs((ff.theta[it]-thetaStep/2)/ts*r)*2+smoothing;
			const r2 = Math.abs((ff.theta[it]+thetaStep/2)/ts*r)*2-smoothing;
			for (let ip = 0; ip < ff.phiPoints; ip++) {
				let a1 = ff.phi[ip] - phiStep/2-smoothing;
				let a2 = ff.phi[ip] + phiStep/2+smoothing;

				if (ff.theta[it] < 0){
					a1 += Math.PI;
					a2 += Math.PI;
				}
				ctx.fillStyle = cmap_vals[ip][it];
				ctx.beginPath();
				if (it == tci && ip == pci){
					ctx.arc(0.0, 0.0, r2, 0, 2*Math.PI);
				}
				else{
					ctx.arc(0.0, 0.0, r2, a1, a2);
					ctx.lineTo(r1*Math.cos(a2), r1*Math.sin(a2));
					ctx.arc(0.0, 0.0, r1, a2, a1, true);
				}
				ctx.closePath();
				ctx.lineWidth = 0.0;
				ctx.fill();
			}
		}
		this.add_phi_grid(this.phiSteps, 1/(this.thetaSteps-1));
		this.add_theta_grid(this.thetaSteps);
	}
	add_phi_grid(steps, startFraction){
		const canvas = this.canvas;
		if (steps === undefined) steps = 13;
		if (startFraction === undefined) startFraction = 0.0;
		const ctx = canvas.getContext('2d');
		const scale = canvas.width*0.5;
		const c = 2*Math.PI/(steps - 1);
		const start = startFraction*scale;
		for (let i = 0; i < (steps-1); i++){
			const ph = i*c
			ctx.beginPath();
			ctx.moveTo(Math.cos(ph)*start, Math.sin(ph)*start);
			ctx.lineTo(Math.cos(ph)*scale, Math.sin(ph)*scale);
			ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
			ctx.lineWidth = 20.0;
			ctx.setLineDash([100, 100]);
			ctx.stroke();
			ctx.closePath();
		}
	}
	add_theta_grid(steps){
		if (steps === undefined) steps = 7;
		const canvas = this.canvas;
		const ctx = canvas.getContext('2d');
		const scale = canvas.width*0.5;
		const c = 1/(steps - 1);
		for (let i = 1; i < (steps-1); i++){
			ctx.beginPath();
			ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
			ctx.lineWidth = 20.0;
			ctx.setLineDash([100, 100]);
			ctx.arc(0.0, 0.0, scale*(i*c), 0, 2*Math.PI);
			ctx.stroke();
			ctx.closePath();
		}
	}
}

export class PlotFarfield2DEngineUV extends PlotFarfield2DEngineABC{
	constructor(parent, uSteps, vSteps){
		super(parent);
		if (uSteps === undefined) uSteps = 11;
		if (vSteps === undefined) vSteps = 11;
		this.vSteps = vSteps;
		this.uSteps = uSteps;
	}
	draw(cmap_vals){
		if (!this.isValid) return;
		const canvas = this.canvas;
		const ctx = canvas.getContext('2d');
		const ff = this.ff;
		ctx.reset();
		canvas.width = CANVAS_SCALER;
		canvas.height = CANVAS_SCALER;
		const r = Math.min(canvas.width/2, canvas.height/2);
		const ur = (ff.u[ff.u.length-1] - ff.u[0])/2;
		const vr = (ff.v[ff.u.length-1] - ff.v[0])/2;
		ctx.translate(canvas.width/2, canvas.height/2);
		ctx.scale(1.0, -1.0);
		const smoothing = 1;
		const du = (ff.u[1] - ff.u[0]);
		const dv = (ff.v[1] - ff.v[0]);
		const uStep = du*r/ur;
		const vStep = dv*r/vr;

		const uWidth = uStep+smoothing*2;
		const vWidth = vStep+smoothing*2;

		canvas.index_from_event = (e) => {
			const rect = canvas.getBoundingClientRect();
			const cu = 2*(e.clientX - rect.left)/rect.width - 1.0;
			const cv = 1-2*(e.clientY - rect.top)/rect.height;
			let iu = Math.round((cu + 1)*ur/du);
			let iv = Math.round((cv + 1)*vr/dv);

			if (iu >= ff.u.length) iu = ff.u.length - 1;
			if (iv >= ff.v.length) iv = ff.v.length - 1;
			if (iu < 0) iu = 0;
			if (iv < 0) iv = 0;
			return [ff.u[iu], ff.v[iv], iu, iv];
		};

		for (let iu = 0; iu < ff.uPoints; iu++) {
			let u1 = iu*uStep - r;
			for (let iv = 0; iv < ff.vPoints; iv++) {
				let v1 = iv*vStep - r;
				ctx.fillStyle = cmap_vals[iv][iu];
				ctx.beginPath();
				ctx.rect(u1-smoothing, v1-smoothing, uWidth, vWidth);
				ctx.closePath();
				ctx.lineWidth = 0.0;
				ctx.fill();
			}
		}
		this.add_u_grid(this.uSteps);
		this.add_v_grid(this.vSteps);
		this.add_border();
	}
	add_border(){
		const canvas = this.canvas;
		const ctx = canvas.getContext('2d');
		const rect = canvas.getBoundingClientRect();
		ctx.save();
		ctx.scale(CANVAS_SCALER, CANVAS_SCALER);
		ctx.scale(1/rect.width, 1/rect.height);

		ctx.beginPath();
		ctx.moveTo(-rect.width/2.0, -rect.height/2.0);
		ctx.lineTo(-rect.width/2.0, rect.height/2.0);
		ctx.lineTo(rect.width/2.0, rect.height/2.0);
		ctx.lineTo(rect.width/2.0, -rect.height/2.0);
		ctx.lineTo(-rect.width/2.0, -rect.height/2.0);
		ctx.strokeStyle = 'rgba(255, 255, 255, 0.4)';
		ctx.setLineDash([0]);
		ctx.lineWidth = 1;
		ctx.stroke();
		ctx.closePath();
	}
	add_u_grid(steps){
		const canvas = this.canvas;
		if (steps === undefined) steps = 11;
		const ctx = canvas.getContext('2d');
		const rect = canvas.getBoundingClientRect();
		ctx.save();
		ctx.scale(CANVAS_SCALER, CANVAS_SCALER);
		ctx.scale(1/rect.width, 1/rect.height);
		const step = rect.width/(steps - 1);
		// scale to bound rectangle which helps reduce aliasing and blur of grid.
		for (let i = 1; i < (steps-1); i++){
			ctx.beginPath();
			let cx = (step*i-rect.width/2).toFixed(0);
			ctx.moveTo(cx, -rect.height/2);
			ctx.lineTo(cx, rect.height/2);
			ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
			ctx.lineWidth = 1;
			ctx.setLineDash([5, 5]);
			ctx.stroke();
			ctx.closePath();
		}
		ctx.restore();
	}
	add_v_grid(steps){
		const canvas = this.canvas;
		if (steps === undefined) steps = 11;
		const ctx = canvas.getContext('2d');
		const rect = canvas.getBoundingClientRect();
		ctx.save();
		ctx.scale(CANVAS_SCALER, CANVAS_SCALER);
		ctx.scale(1/rect.width, 1/rect.height);
		const step = rect.height/(steps - 1);
		// scale to bound rectangle which helps reduce aliasing and blur of grid.
		for (let i = 1; i < (steps-1); i++){
			ctx.beginPath();
			let cy = (step*i-rect.height/2).toFixed(0);
			ctx.moveTo(-rect.width/2.0, cy);
			ctx.lineTo(rect.width/2.0, cy);
			ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
			ctx.lineWidth = 1;
			ctx.setLineDash([5, 5]);
			ctx.stroke();
			ctx.closePath();
		}
		ctx.restore();
	}
}

export class PlotFarfield2DEngineLudwig3 extends PlotFarfield2DEngineABC{
	constructor(parent, azSteps, elSteps){
		super(parent);
		if (azSteps === undefined) azSteps = 13;
		if (elSteps === undefined) elSteps = 13;
		this.elSteps = elSteps;
		this.azSteps = azSteps;
	}
	draw(cmap_vals){
		if (!this.isValid) return;
		const canvas = this.canvas;
		const ctx = canvas.getContext('2d');
		const ff = this.ff;
		ctx.reset();
		canvas.width = CANVAS_SCALER;
		canvas.height = CANVAS_SCALER;
		const r = Math.min(canvas.width/2, canvas.height/2);
		const ur = (ff.az[ff.az.length-1] - ff.az[0])/2;
		const vr = (ff.el[ff.el.length-1] - ff.el[0])/2;
		ctx.translate(canvas.width/2, canvas.height/2);
		ctx.scale(1.0, -1.0);
		const smoothing = 1;
		const uStep = (ff.az[1] - ff.az[0])*r/ur;
		const vStep = (ff.el[1] - ff.el[0])*r/vr;

		const uWidth = uStep+smoothing*2;
		const vWidth = vStep+smoothing*2;

		canvas.index_from_event = (e) => {
			const rect = canvas.getBoundingClientRect();
			const cu = 2*(e.clientX - rect.left)/rect.width - 1.0;
			const cv = 1-2*(e.clientY - rect.top)/rect.height;
			let iu = Math.round((cu + 1)*ur/(ff.az[1] - ff.az[0]));
			let iv = Math.round((cv + 1)*vr/(ff.el[1] - ff.el[0]));

			if (iu >= ff.az.length) iu = ff.az.length - 1;
			if (iv >= ff.el.length) iv = ff.el.length - 1;
			if (iu < 0) iu = 0;
			if (iv < 0) iv = 0;

			return [ff.az[iu]*180/Math.PI, ff.el[iv]*180/Math.PI, iu, iv];
		};

		for (let iu = 0; iu < ff.azPoints; iu++) {
			let u1 = iu*uStep - r;
			for (let iv = 0; iv < ff.elPoints; iv++) {
				let v1 = iv*vStep - r;
				ctx.fillStyle = cmap_vals[iv][iu];
				ctx.beginPath();
				ctx.rect(u1-smoothing, v1-smoothing, uWidth, vWidth);
				ctx.closePath();
				ctx.lineWidth = 0.0;
				ctx.fill();
			}
		}
		this.add_az_grid(this.azSteps);
		this.add_el_grid(this.elSteps);
		this.add_border();
	}
	add_border(){
		const canvas = this.canvas;
		const ctx = canvas.getContext('2d');
		const rect = canvas.getBoundingClientRect();
		ctx.save();
		ctx.scale(CANVAS_SCALER, CANVAS_SCALER);
		ctx.scale(1/rect.width, 1/rect.height);

		ctx.beginPath();
		ctx.moveTo(-rect.width/2.0, -rect.height/2.0);
		ctx.lineTo(-rect.width/2.0, rect.height/2.0);
		ctx.lineTo(rect.width/2.0, rect.height/2.0);
		ctx.lineTo(rect.width/2.0, -rect.height/2.0);
		ctx.lineTo(-rect.width/2.0, -rect.height/2.0);
		ctx.strokeStyle = 'rgba(255, 255, 255, 0.4)';
		ctx.setLineDash([0]);
		ctx.lineWidth = 1;
		ctx.stroke();
		ctx.closePath();
	}
	add_az_grid(steps){
		const canvas = this.canvas;
		if (steps === undefined) steps = 13;
		const ctx = canvas.getContext('2d');
		const rect = canvas.getBoundingClientRect();
		ctx.save();
		ctx.scale(CANVAS_SCALER, CANVAS_SCALER);
		ctx.scale(1/rect.width, 1/rect.height);
		const step = rect.width/(steps - 1);
		// scale to bound rectangle which helps reduce aliasing and blur of grid.
		for (let i = 1; i < (steps-1); i++){
			ctx.beginPath();
			let cx = (step*i-rect.width/2).toFixed(0);
			ctx.moveTo(cx, -rect.height/2);
			ctx.lineTo(cx, rect.height/2);
			ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
			ctx.lineWidth = 1;
			ctx.setLineDash([5, 5]);
			ctx.stroke();
			ctx.closePath();
		}
		ctx.restore();
	}
	add_el_grid(steps){
		const canvas = this.canvas;
		if (steps === undefined) steps = 13;
		const ctx = canvas.getContext('2d');
		const rect = canvas.getBoundingClientRect();
		ctx.save();
		ctx.scale(CANVAS_SCALER, CANVAS_SCALER);
		ctx.scale(1/rect.width, 1/rect.height);
		const step = rect.height/(steps - 1);
		// scale to bound rectangle which helps reduce aliasing and blur of grid.
		for (let i = 1; i < (steps-1); i++){
			ctx.beginPath();
			let cy = (step*i-rect.height/2).toFixed(0);
			ctx.moveTo(-rect.width/2.0, cy);
			ctx.lineTo(rect.width/2.0, cy);
			ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
			ctx.lineWidth = 1;
			ctx.setLineDash([5, 5]);
			ctx.stroke();
			ctx.closePath();
		}
		ctx.restore();
	}
}
