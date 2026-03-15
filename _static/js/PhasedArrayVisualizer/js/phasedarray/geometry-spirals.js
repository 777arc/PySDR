import { Geometry } from "./geometry.js";

export class ArchimedianSpiral extends Geometry{
	static title = 'Archimedian Spiral';
	static args = ['r0', 'r-max', 'spirals', 'count-per-spiral', 'total-rotation'];
	static controls = {
		'geometry': {'title': null},
		'r0': {'title': "R-Start (λ)", 'type': "float", 'min': 0.0, 'default': 0.5},
		'r-max': {'title': "R-Max (λ)", 'type': "float", 'min': 0.0, 'default': 2.5},
		'spirals': {'title': 'Spirals', 'type': "int", 'default': 1, 'min': 1},
		'count-per-spiral': {'title': 'Elements per Spiral', 'type': "int", 'default': 31, 'min': 1},
		'total-rotation': {'title': 'Total Rotations (deg)', 'type': "float", 'default': 990, 'min': 0},
	};
	constructor(r0, rmax, spirals, count, rotations){
		super();

		if (r0 <= 0) r0 = 0.0
		if (rmax <= 0) rmax = 0.5

		this.r0 = Number(r0);
		this.rmax = Number(rmax);
		if (spirals <= 0) spirals = 1;
		this.spirals = Number(spirals);

		this.rotations = Math.max(0, Number(rotations))*Math.PI/180;
		this.count = Number(count);
	}
	build(){
		let x = []
		let y = []
		let sc = (this.rmax - this.r0)/this.rotations;
		let rc = this.rotations/(this.count - 1);

		let index = 0;
		if (this.spirals > 1){
			x.push(0.0);
			y.push(0.0);
			index = 1;
		}

		for (let j = 0; j < this.spirals; j++){
			const io = 2*Math.PI/this.spirals*j;
			for (let i = 0; i < this.count; i++){
				const phi = i*rc;
				const r = this.r0 + sc*phi;
				x.push(r*Math.cos(phi + io));
				y.push(r*Math.sin(phi + io));
			}
		}
		this.set_xy(Float32Array.from(x), Float32Array.from(y));
		this.auto_compute_dx_dy(index);
	}
}

export class DoughertyLogSpiral extends Geometry{
	static title = 'Dougherty Log-Spiral';
	static args = ['r0', 'r-max', 'spirals', 'count-per-spiral', 'offset-rotation'];
	static controls = {
		'geometry': {'title': null},
		'r0': {'title': "R-Start (λ)", 'type': "float", 'min': 0.01, 'default': 0.5},
		'r-max': {'title': "R-Max (λ)", 'type': "float", 'min': 0.0, 'default': 2.5},
		'spirals': {'title': 'Spirals', 'type': "int", 'default': 1, 'min': 1},
		'count-per-spiral': {'title': 'Elements per Spiral', 'type': "int", 'default': 31, 'min': 1},
		'offset-rotation': {'title': 'Angle (deg)', 'type': "float", 'default': 84, 'min': 0},
	};
	constructor(r0, rmax, spirals, count, rotations){
		super();

		if (r0 <= 0) r0 = 0.5
		if (rmax <= 0) rmax = 2.0

		this.r0 = Number(r0);
		this.rmax = Number(rmax);

		if (spirals <= 0) spirals = 1;
		this.spirals = Number(spirals);

		this.rotations = Math.max(0, Number(rotations))*Math.PI/180;
		this.count = Number(count);
	}
	build(){
		const cov = 1/Math.tan(this.rotations);
		const sc = 1/(this.count-1)*(this.rmax/this.r0 - 1);

		let x = []
		let y = []
		let index = 0;
		if (this.spirals > 1){
			x.push(0.0);
			y.push(0.0);
			index = 1;
		}

		let phi, r;
		for (let j = 0; j < this.spirals; j++){
			const io = 2*Math.PI/this.spirals*j;
			for (let i = 0; i < this.count; i++){
				if (this.rotations == 0){
					phi = 0;
					r = this.r0 + i*(this.rmax-this.r0)/(this.count-1);
				}
				else{
					phi = 1/cov*Math.log(1 + i*sc);
					r = this.r0*Math.exp(cov*phi);
				}

				console.log(r, phi, Math.exp(cov*phi), phi, cov);
				x.push(r*Math.cos(phi + io));
				y.push(r*Math.sin(phi + io));
			}
		}
		this.set_xy(Float32Array.from(x), Float32Array.from(y));
		this.auto_compute_dx_dy(index);
	}
}

export class ArcondoulisSpiral extends Geometry{
	static title = 'Arcondoulis Spiral';
	static args = ['r0', 'r-max', 'spirals', 'count-per-spiral', 'total-rotation', 'eta-x', 'eta-y'];
	static controls = {
		'geometry': {'title': null},
		'r0': {'title': "R-Start (λ)", 'type': "float", 'min': 0.01, 'default': 0.5, 'step': 0.1},
		'r-max': {'title': "R-Max (λ)", 'type': "float", 'min': 0.0, 'default': 2.5, 'step': 0.1},
		'spirals': {'title': 'Spirals', 'type': "int", 'default': 1, 'min': 1},
		'count-per-spiral': {'title': 'Elements per Spiral', 'type': "int", 'default': 31, 'min': 1},
		'total-rotation': {'title': 'Total Rotations (deg)', 'type': "float", 'default': 990, 'min': 0},
		'eta-x': {'title': 'Squashing-X', 'type': "float", 'default': 0.9, 'min': 0, 'step': 0.1},
		'eta-y': {'title': 'Squashing-X', 'type': "float", 'default': 0.9, 'min': 0, 'step': 0.1},
	};
	constructor(r0, rmax, spirals, count, rotations, ex, ey){
		super();

		if (r0 <= 0) r0 = 0.5
		if (rmax <= 0) rmax = 2.0
		if (ex <= 0) ex = 0.9
		if (ey <= 0) ey = 0.9
		if (spirals <= 0) spirals = 1;
		this.spirals = Number(spirals);

		this.r0 = Number(r0);
		this.rmax = Number(rmax);
		this.ex = Number(ex);
		this.ey = Number(ey);

		this.rotations = Math.max(0, Number(rotations))*Math.PI/180;
		this.count = Number(count);
	}
	build(){
		const a = this.r0*(this.count/(this.ex*this.count + 1));
		const b = 1/this.rotations*Math.log(this.rmax/(a*Math.sqrt((1 + this.ex)**2*Math.cos(this.rotations)**2+(1 + this.ey)**2*Math.sin(this.rotations)**2)));

		let x = []
		let y = []
		const sc = this.rotations/(this.count - 1);
		let index = 0;
		if (this.spirals > 1){
			x.push(0.0);
			y.push(0.0);
			index = 1;
		}
		for (let j = 0; j < this.spirals; j++){
			const io = 2*Math.PI/this.spirals*j;
			for (let i = 0; i < this.count; i++){
				const phi = i*sc;
				const m = a*Math.exp(b*phi);
				x.push((i + 1 + this.ex*this.count)/this.count*Math.cos(phi + io)*m);
				y.push((i + 1 + this.ey*this.count)/this.count*Math.sin(phi + io)*m);
			}
		}
		this.set_xy(Float32Array.from(x), Float32Array.from(y));
		this.auto_compute_dx_dy(index);
	}
}
