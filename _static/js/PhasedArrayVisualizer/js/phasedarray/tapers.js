/**
* This file contains a bunch of taper functions.
*
* Many of these functions are from:
* https://www.researchgate.net/publication/316281181_Catalog_of_Window_Taper_Functions_for_Sidelobe_Control
* */
import {bessel_modified_0, linspace, normalize, ones} from "../util.js";

export class Uniform{
	static title = 'Uniform';
	static args = [];
	static controls = {};
	calculate_weights(x){ return ones(x.length); }
	normalize_from_geometry(x, dx){ return normalize(x); }
	normalize_from_radial_geometry(x, y, dx, dy){
		const mr = Float32Array.from(x, (ix, i) => Math.sqrt(ix**2 + y[i]**2));
		const maxR = Math.max(...mr);
		return Float32Array.from(mr, (v) => v/maxR*0.5);
	}
	calculate_from_geometry(x, dx){ return this.calculate_weights(this.normalize_from_geometry(x, dx)); }
	calculate_from_radial_geometry(x, y, dx, dy){ return this.calculate_weights(this.normalize_from_radial_geometry(x, y, dx, dy)); }
}

export class TrianglePedestal extends Uniform{
	static title = 'Triangle on a Pedestal';
	static args = ['taper-par-1'];
	static controls = {
		...Uniform.controls,
		'taper-par-1': {
			'title': "Pedestal",
			'type': "float",
			'default': 0.0,
			'min': 0.0,
			'max': 1.0,
			'step': 0.1
		},
	};
	constructor(pedestal){
		super();
		this.pedestal = Math.min(1, Math.max(0, Math.abs(pedestal)));
	}
	calculate_weights(x){
		const maxX = Math.max(...x);
		const minX = Math.min(...x);
		const den = (maxX - minX)/2.0;
		const sc = 1 - this.pedestal;
		return Float32Array.from(x, (e) => 1 - sc*Math.abs((e - minX)/den - 1.0));
	}
}

export class TaylorNBar extends Uniform{
	static title = 'Taylor N-Bar';
	static args = ['taper-par-1', 'taper-par-2'];
	static controls = {
		...Uniform.controls,
		'taper-par-1': {
			'title': "N Bar",
			'type': "int",
			'default': 5,
			'min': 1
		},
		'taper-par-2': {
			'title': "SLL",
			'type': "float",
			'default': 25.0,
			'min': 13.0
		}
	};
	constructor(nbar, sll){
		super();
		this.nbar = nbar;
		this.sll = Math.max(13, Math.abs(sll));
	}
	normalize_from_geometry(x, dx){
		const maxX = Math.max(...x) + dx/2;
		const minX = Math.min(...x) - dx/2;
		const den = (maxX - minX);
		return Float32Array.from(x, (xi) => (xi - minX)/den - 0.5);
	}
	normalize_from_radial_geometry(x, y, dx, dy){
		const maxX = Math.max(...x);
		const minX = Math.min(...x);
		const maxY = Math.max(...y);
		const minY = Math.min(...y);

		const wx = (maxX - minX);
		const wy = (maxY - minY);
		const ms = Math.sqrt(wx**2 + wy**2);
		const dr = dx*wx/ms + dy*wy/ms;

		const ox = minX + wx/2
		const oy = minY + wy/2

		const mr = Float32Array.from(x, (xi, i) => Math.sqrt((xi - ox)**2 + (y[i] - oy)**2));
		const maxR = Math.max(...mr) + dr/2;
		return Float32Array.from(mr, (ri) => ri/maxR*0.5);
	}
	calculate_weights(x){
		const nbar = this.nbar;
		const sll = this.sll;
		if (nbar == 1 && sll == 13) return ones(x.length);
		const nu = 10**(this.sll/20.);
		const A = Math.acosh(nu)/Math.PI;
		const A2 = A**2;
		const sigma2 = nbar**2/(A2 + (nbar - 0.5)**2);
		const Fm = [];

		const _f = (m) => {
			const c1 = (m**2/sigma2);
			let f1 = 1.0;
			for (let n = 1; n < nbar; n++){
				f1 *= (1 - c1/(A2 + (n - 0.5)**2));
			}
			let f2 = 1.0;
			for (let n = 1; n < nbar; n++){
				if (n == m) continue;
				f2 *= (1 - m**2/n**2);
			}
			return -1*((-1)**m/2)*f1/f2;
		}
		for (let m = 1; m < nbar; m++) Fm.push(_f(m));
		const pi2 = 2*Math.PI;
		return Float32Array.from(x, (e) => {
			let a1 = 0.0;
			for (let m = 0; m < Fm.length; m++){
				a1 += Fm[m]*Math.cos(pi2*(m + 1)*e);
			}
			return 1 + 2*a1;
		});
	}
}

export class TaylorModified extends TaylorNBar{
	static title = 'Taylor Modified';
	static args = ['taper-par-1'];
	static controls = {
		...Uniform.controls,
		'taper-par-1': {
			'title': "SLL",
			'type': "float",
			'default': 25.0,
			'min': 13.0
		}
	};
	constructor(sll){
		super();
		this.sll = Math.max(13, Math.abs(sll));
	}
	calculate_weights(x){
		const s = 10**(this.sll/20.0)/4.60333;
		if (s <= 1) return ones(x.length);
		const err = 1e-6;

		let B = linspace(0.00001, 10, 101);
		let counter = 0;
		while (1){
			if (counter > 50) {
				B = B[0];
				break;
			}
			counter++;
			let v = Float32Array.from(B, (o) => Math.sinh(Math.PI*o)/(Math.PI*o));
			let e = Float32Array.from(v, (o) => Math.abs(o - s));

			let minE = Infinity;
			let minI = 0;
			for (let i = 0; i < e.length; i++){
				if (e[i] < minE){
					minE = e[i];
					minI = i;
				}
			}
			if (minE < err) {
				B = B[minI];
				break;
			}
			let o1, o2;
			if (minI == 0) o1 = 1e-9;
			else o1 = B[minI - 1];
			if (minI == v.length - 1) o2 = B[B.length-1] + 1.0;
			else o2 = B[minI + 1];
			B = linspace(o1, o2, 11);
		}
		return Float32Array.from(x, (e) => bessel_modified_0(Math.PI*B*Math.sqrt(1 - (2*e)**2)));
	}
}

export class Parzen extends Uniform{
	static title = 'Parzen';
	static args = [];
	calculate_weights(x){
		const s = 8/3;
		return Float32Array.from(x, (e) => {
			const t = Math.abs(e);
			if (t <= 0.25) return s*(1 - 24*t**2 + 48*t**3);
			else if (t <= 0.5) return s*(2 - 12*t + 24*t**2 - 16*t**3);
			return 0.0;
		});
	}
}

export class ParzenAlgebraic extends Uniform{
	static title = 'Parzen Algebraic';
	static args = ['taper-par-1', 'taper-par-1'];
	static controls = {
		...Uniform.controls,
		'taper-par-1': {
			'title': "Gamma",
			'type': "float",
			'default': 1.0,
			'min': 0.0,
			'max': 1.0,
			'step': 0.1
		},
		'taper-par-2': {
			'title': "u",
			'type': "float",
			'default': 2.0,
			'min': 0.0,
			'step': 0.1
		},
	};
	constructor(gamma, u){
		super();
		this.gamma = Math.max(0.001, Math.min(1, Math.abs(gamma)));
		this.u = Math.max(0.001, u);
	}
	calculate_weights(x){
		const g = this.gamma;
		const u = this.u;
		// ignore scale because it's normalized out.
		//const A = 1/(1 - g/(1 + u))
		return Float32Array.from(x, (e) => 1 - g*Math.abs(2*e)**u);
	}
}

export class Welch extends Uniform{
	static title = 'Welch';
	static args = [];
	calculate_weights(x){
		return Float32Array.from(x, (e) => 3/2*(1 - 4*e**2));
	}
}

export class Connes extends Uniform{
	static title = 'Connes';
	static args = ['taper-par-1'];
	static controls = {
		...Uniform.controls,
		'taper-par-1': {
			'title': "Alpha",
			'type': "float",
			'default': 1.0,
			'min': 0.0,
			'step': 0.1
		},
	};
	constructor(alpha){
		super();
		this.alpha = Math.max(0.001, Math.abs(alpha));
	}
	calculate_weights(x){
		const a2 = this.alpha**2;
		// ignoring scale factors because they get normalized out.
		//const a4 = this.alpha**4;
		//const A = 15*a4/(3 - 10*a2 + 15*a4);
		return Float32Array.from(x, (e) => (a2 - 4*(e**2))**2);
	}
}

export class SinglaSingh extends Uniform{
	static title = 'Singla-Singh (order 1)';
	static args = [];
	calculate_weights(x){
		return Float32Array.from(x, (e) => 1-4*e**2*(3 - 4*Math.abs(e)));
	}
}

export class Lanczos extends Uniform{
	static title = 'Lanczos';
	static args = ['taper-par-1'];
	static controls = {
		...Uniform.controls,
		'taper-par-1': {
			'title': "L",
			'type': "float",
			'default': 2.0,
			'min': 0.0,
			'step': 0.1
		},
	};
	constructor(l){
		super();
		this.l = Math.max(0.001, Math.abs(l));
	}
	calculate_weights(x){
		return Float32Array.from(x, (e) => {
			if (e == 0) return 1.0;
			const v = 2*Math.PI*e;
			return (Math.sin(v)/v)**this.l;
		});
	}
}

export class SincLobe extends Lanczos{
	static title = 'Sinc Lobe';
	static args = [];
	static controls = Uniform.controls;
	constructor(){ super(1.0); }
}

export class Fejer extends Lanczos{
	static title = 'Fejér';
	static args = [];
	static controls = Uniform.controls;
	constructor(){ super(2.0); }
}

export class delaVallePoussin extends Lanczos{
	static title = 'de la Vallée Poussin';
	static args = [];
	static controls = Uniform.controls;
	constructor(){ super(4.0); }
}

export class RaisedCosine extends Uniform{
	static title = 'Raised Cosine';
	static args = ['taper-par-1'];
	static controls = {
		...Uniform.controls,
		'taper-par-1': {
			'title': "Pedestal",
			'type': "float",
			'default': 0.75,
			'min': 0.0,
			'max': 1.0,
			'step': 0.05
		},
	};
	constructor(alpha){
		super();
		this.alpha = Math.max(0.5, Math.min(1.0, Math.abs(alpha)));
	}
	calculate_weights(x){
		const alpha = this.alpha;
		const s = (1 - alpha)/alpha;
		const pi2 = 2*Math.PI
		return Float32Array.from(x, (e) => 1 + s*Math.cos(pi2*e));
	}
}

export class Hamming extends RaisedCosine{
	static title = 'Hamming';
	static args = [];
	static controls = Uniform.controls;
	constructor(){ super(25/46); }
}

export class Hann extends RaisedCosine{
	static title = 'Hann';
	static args = [];
	static controls = Uniform.controls;
	constructor(){ super(0.5); }
}

export class GeneralizedHamming extends Uniform{
	static title = 'Generalized Hamming';
	static args = ['taper-par-1'];
	static controls = {
		...Uniform.controls,
		'taper-par-1': {
			'title': "v",
			'type': "float",
			'default': 0.0,
			'min': -0.5,
			'step': 0.1
		},
	};
	constructor(v){
		super();
		this.v = Math.max(-0.5, v);
	}
	calculate_weights(x){
		const v = this.v;
		const alpha = (2 + 3*v + v**2)/(23 + 9*v + v**2);
		return Float32Array.from(x, (e) => {
			const ep = e*Math.PI;
			return alpha*Math.cos(ep)**v + (1 - alpha)*Math.cos(ep)**(v + 2);
		});
	}
}

export class RaisedPowerofCosine extends Uniform{
	static title = 'Raised Power-of-Cosine';
	static args = ['taper-par-1', 'taper-par-2'];
	static controls = {
		...Uniform.controls,
		'taper-par-1': {
			'title': "Power",
			'type': "float",
			'default': 1.0,
			'min': 0,
			'step': 0.1
		},
		'taper-par-2': {
			'title': "Pedestal",
			'type': "float",
			'default': 0.0,
			'min': 0,
			'max': 1.0,
			'step': 0.1
		},
	};
	constructor(m, pedestal){
		super();
		this.m = Math.max(0, m);
		this.pedestal = Math.min(1.0, Math.max(0, pedestal));
	}
	calculate_weights(x){
		const a = this.pedestal;
		const a0 = 1 - a;
		return Float32Array.from(x, (e) => a + a0*Math.cos(e*Math.PI)**this.m);
	}
}

export class ParzenCosine extends Uniform{
	static title = 'Parzen Cosine';
	static args = ['taper-par-1', 'taper-par-2'];
	static controls = {
		...Uniform.controls,
		'taper-par-1': {
			'title': "Gamma",
			'type': "float",
			'default': 0.5,
			'min': 0.0,
			'max': 1.0,
			'step': 0.1
		},
		'taper-par-2': {
			'title': "Power",
			'type': "float",
			'default': 1.0,
			'min': 0,
			'step': 0.1
		},
	};
	constructor(gamma, m){
		super();
		this.m = Math.max(0, m);
		this.gamma = Math.min(1.0, Math.max(0, gamma));
	}
	calculate_weights(x){
		const g = this.gamma;
		const m = this.m;
		const gpi = g*Math.PI;
		return Float32Array.from(x, (e) => 1+Math.cos(gpi*Math.abs(2*e)**m));
	}
}

export class ParzenGeometric extends Uniform{
	static title = 'Parzen Geometric';
	static args = ['taper-par-1', 'taper-par-2'];
	static controls = {
		...Uniform.controls,
		'taper-par-1': {
			'title': "Alpha",
			'type': "float",
			'default': 1.5,
			'min': 0.0,
			'step': 0.1
		},
		'taper-par-2': {
			'title': "Power",
			'type': "float",
			'default': 3.0,
			'min': 0,
			'step': 0.1
		},
	};
	constructor(alpha, r){
		super();
		this.r = Math.max(0, Math.abs(r));
		this.alpha = Math.max(0, Math.abs(alpha));
	}
	calculate_weights(x){
		const a2 = 2*this.alpha;
		const r = this.r;
		return Float32Array.from(x, (e) => 1/(1 + Math.abs(a2*e)**r));
	}
}

export class Bohman extends Uniform{
	static title = 'Parzen Cosine';
	calculate_weights(x){
		const p24 = Math.PI**2/4;
		const p4 = Math.PI/4;
		const p2 = 2*Math.PI;
		return Float32Array.from(x, (e) => {
			const t = Math.abs(e);
			const pit = p2*t;
			return p24*(1 - 2*t)*Math.cos(pit) + p4*Math.sin(pit)
		});
	}
}

export class Trapezoid extends Uniform{
	static title = 'Trapezoid';
	static args = ['taper-par-1', 'taper-par-2'];
	static controls = {
		...Uniform.controls,
		'taper-par-1': {
			'title': "Offset",
			'type': "float",
			'default': 0.1,
			'min': 0.0,
			'max': 0.5,
			'step': 0.1,
			'desc': "Offset from center when trapezoid starts."
		},
		'taper-par-2': {
			'title': "Pedestal",
			'type': "float",
			'default': 0.0,
			'min': 0,
			'max': 1.0,
			'step': 0.1,
		}
	};
	constructor(offset, pedestal){
		super();
		this.offset = Math.min(0.5, Math.max(0, offset));
		this.pedestal = Math.min(1.0, Math.max(0.0, Math.abs(pedestal)));
	}
	calculate_weights(x){
		const p = this.pedestal;
		const a = this.offset;
		const s = (1 - p)/(0.5 - a);
		return Float32Array.from(x, (e) => {
			const t = Math.abs(e);
			if (t <= a) return 1.0;
			if (t <= 0.5) return (0.5 - t)*s + p;
			return p
		});
	}
}

export class Tukey extends Uniform{
	static title = 'Tukey';
	static args = ['taper-par-1'];
	static controls = {
		...Uniform.controls,
		'taper-par-1': {
			'title': "Offset",
			'type': "float",
			'default': 0.1,
			'min': 0.0,
			'max': 0.5,
			'step': 0.1,
			'desc': "Offset from center when curve starts."
		},
	};
	calculate_weights(x){
		const a = this.offset;
		if (a >= 0.5) return ones(x.length);

		const sc = 2/(0.5 - a);
		const ad1 = 1/(0.5 - a);
		const ad2 = 1/(0.5 - a);
		return Float32Array.from(x, (e) => {
			const t = Math.abs(e);
			if (t <= a) return sc;
			if (t <= 0.5) return (1 + Math.cos(Math.PI*(t - a)*ad1))*ad2;
			return 0.0
		});
	}
}

export class BartlettHann extends Uniform{
	static title = 'Bartlett-Hann';
	calculate_weights(x){
		const a0 = 0.62;
		const a1 = 0.48;
		const a2 = 0.38;
		const pi2 = 2*Math.PI;
		return Float32Array.from(x, (e) => 2*(a0 - a1*Math.abs(e) + a2*Math.cos(pi2*e)));
	}
}

export class Blackman extends Uniform{
	static title = 'Blackman';
	calculate_weights(x){
		const pi2 = 2*Math.PI;
		const pi4 = 4*Math.PI;
		const a10 = 9240/7938;
		const a20 = 1430/7938;
		return Float32Array.from(x, (e) => (1 + a10*Math.cos(pi2*e) + a20*Math.cos(pi4*e)));
	}
}

export class Exponential extends Uniform{
	static title = 'Exponential';
	static args = ['taper-par-1'];
	static controls = {
		...Uniform.controls,
		'taper-par-1': {
			'title': "Alpha",
			'type': "float",
			'default': 2.0,
			'min': 0,
			'step': 0.1,
			'desc': "Decay parameter"
		},
	};
	constructor(alpha){
		super();
		this.alpha = Math.max(0.0, alpha);
	}
	calculate_weights(x){
		const a2 = -2*this.alpha;
		return Float32Array.from(x, (e) => Math.exp(a2*Math.abs(e)));
	}
}

export class HanningPoisson extends Uniform{
	static title = 'Hanning-Poisson';
	static args = ['taper-par-1'];
	static controls = {
		...Uniform.controls,
		'taper-par-1': {
			'title': "Alpha",
			'type': "float",
			'default': 0.5,
			'min': 0,
			'step': 0.1,
			'desc': "Decay parameter."
		},
	};
	constructor(alpha){
		super();
		this.alpha = Math.max(0.0, alpha);
	}
	calculate_weights(x){
		const a2 = -2*this.alpha;
		const pi2 = 2*Math.PI;
		return Float32Array.from(x, (e) => Math.exp(a2*Math.abs(e))*(1 + Math.cos(pi2*e)));
	}
}

export class Gaussian extends Uniform{
	static title = 'Gaussian';
	static args = ['taper-par-1'];
	static controls = {
		...Uniform.controls,
		'taper-par-1': {
			'title': "Alpha",
			'type': "float",
			'default': 2.0,
			'min': 0,
			'step': 0.1,
			'desc': "Number of standard deviations at which truncation occurs."
		},
	};
	constructor(alpha){
		super();
		this.alpha = Math.max(0.0, alpha);
	}
	calculate_weights(x){
		const a2 = -2*this.alpha**2;
		return Float32Array.from(x, (e) => Math.exp(a2*e**2));
	}
}

export class ParzenExponential extends Uniform{
	static title = 'Parzen Exponential';
	static args = ['taper-par-1', 'taper-par-2'];
	static controls = {
		...Uniform.controls,
		'taper-par-1': {
			'title': "Alpha",
			'type': "float",
			'default': 1.5,
			'min': 0,
			'step': 0.1,
			'desc': "Decay parameter."
		},
		'taper-par-2': {
			'title': "r",
			'type': "float",
			'default': 3.0,
			'min': 0,
			'step': 0.1,
			'desc': "Exponential power."
		},
	};
	constructor(alpha, r){
		super();
		this.alpha = Math.max(0.0, alpha);
		this.r = Math.max(0.0, r);
	}
	calculate_weights(x){
		const a2 = 2*this.alpha;
		const r = this.r
		return Float32Array.from(x, (e) => Math.exp(-1*Math.abs(a2*e)**r));
	}
}

export class DolphChebyshev extends Uniform{
	static title = 'Dolph-Chebyshev';
	static args = ['taper-par-1'];
	static controls = {
		...Uniform.controls,
		'taper-par-1': {
			'title': "SLL",
			'type': "float",
			'default': 30,
			'min': 13,
			'desc': "Desired sidelobe level."
		},
	};
	constructor(sll){
		super();
		this.sll = Math.sll(13, Math.abs(sll));
	}
	calculate_weights(x){
		// TODO: Come back to and finish.
		throw Error("in work.");
		const nu = 10**(this.sll/20.0);
		return Float32Array.from(x, (e) => Math.exp(a2*e**2));
	}
}

export class Cauchy extends Uniform{
	static title = 'Cauchy';
	static args = ['taper-par-1'];
	static controls = {
		...Uniform.controls,
		'taper-par-1': {
			'title': "Alpha",
			'type': "float",
			'default': 3.0,
			'min': 0,
			'step': 0.1,
			'desc': "Decay parameter."
		},
	};
	constructor(alpha){
		super();
		this.alpha = Math.max(0.0, alpha);
	}
	calculate_weights(x){
		const a2 = 2*this.alpha;
		return Float32Array.from(x, (e) => 1/(1 + (a2 * e)**2));
	}
}

export class KaiserBessel extends Uniform{
	static title = 'Kaiser Bessel';
	static args = ['taper-par-1'];
	static controls = {
		...Uniform.controls,
		'taper-par-1': {
			'title': "Alpha",
			'type': "float",
			'default': 1.25,
			'min': 0,
			'step': 0.05,
		},
	};
	constructor(alpha){
		super();
		this.alpha = Math.max(0.0, alpha);
	}
	calculate_weights(x){
		const api = Math.PI*this.alpha;
		return Float32Array.from(x, (e) => bessel_modified_0(api*Math.sqrt(1 - (2*e)**2)));
	}
}

export class Cosh extends Uniform{
	static title = 'Cosh';
	static args = ['taper-par-1'];
	static controls = {
		...Uniform.controls,
		'taper-par-1': {
			'title': "Alpha",
			'type': "float",
			'default': 1.25,
			'min': 0,
			'step': 0.05,
		},
	};
	constructor(alpha){
		super();
		this.alpha = Math.max(0.0, alpha);
	}
	calculate_weights(x){
		const api = Math.PI*this.alpha;
		return Float32Array.from(x, (e) => Math.cosh(api*Math.sqrt(1 - (2*e)**2)));
	}
}

export class AvciNacaroglu extends Uniform{
	static title = 'Avci-Nacaroglu Exponential';
	static args = ['taper-par-1'];
	static controls = {
		...Uniform.controls,
		'taper-par-1': {
			'title': "Alpha",
			'type': "float",
			'default': 1.25,
			'min': 0,
			'step': 0.05,
		},
	};
	constructor(alpha){
		super();
		this.alpha = Math.max(0.0, alpha);
	}
	calculate_weights(x){
		const api = Math.PI*this.alpha;
		return Float32Array.from(x, (e) => Math.exp(api*(Math.sqrt(1 - (2*e)**2) - 1)));
	}
}

export class Knab extends Uniform{
	static title = 'Knab';
	static args = ['taper-par-1'];
	static controls = {
		...Uniform.controls,
		'taper-par-1': {
			'title': "Alpha",
			'type': "float",
			'default': 1.5,
			'min': 0,
			'step': 0.05,
		},
	};
	constructor(alpha){
		super();
		this.alpha = Math.max(0.0, alpha);
	}
	calculate_weights(x){
		const api = Math.PI*this.alpha;
		return Float32Array.from(x, (e) => {
			let d = Math.sqrt(1 - (2*e)**2);
			if (d == 0) return 0.0;
			return Math.sinh(api*d)/d;
		});
	}
}

export class Vorbis extends Uniform{
	static title = 'Vorbis';
	calculate_weights(x){
		const p2 = Math.PI/2;
		return Float32Array.from(x, (e) => Math.sin(p2*Math.cos(Math.PI*e)**2));
	}
}

export class FlatTop5 extends Uniform{
	static title = 'Flat-Top L=5';
	calculate_weights(x){
		const p2 = Math.PI*2;
		const a = [
			0.21557895,
			0.41663158,
			0.277263158,
			0.083578947,
			0.006947368
		]
		return Float32Array.from(x, (e) => {
			let s = 0;
			for (let i = 0; i < 5; i++) s += a[i]*Math.cos(p2*i*e)
			return s;
		});
	}
}

export class FlatTop3 extends Uniform{
	static title = 'Flat-Top L=3';
	calculate_weights(x){
		const p2 = Math.PI*2;
		const a = [
			0.2811,
			0.5209,
			0.1980
		]
		return Float32Array.from(x, (e) => {
			let s = 0;
			for (let i = 0; i < 3; i++) s += a[i]*Math.cos(p2*i*e)
			return s;
		});
	}
}

export const Tapers = [
	Uniform,
	TaylorNBar,
	TaylorModified,
	//RaisedCosine,
	RaisedPowerofCosine,
	//TrianglePedestal,
	Trapezoid,
	Exponential,
	Gaussian,
	KaiserBessel,
	Cauchy,
	Cosh,
	Tukey,
	Hamming,
	GeneralizedHamming,
	Hann,
	HanningPoisson,
	Parzen,
	Connes,
	Welch,
	ParzenAlgebraic,
	ParzenCosine,
	ParzenExponential,
	ParzenGeometric,
	SinglaSingh,
	Lanczos,
	SincLobe,
	Fejer,
	delaVallePoussin,
	Bohman,
	BartlettHann,
	Blackman,
	AvciNacaroglu,
	Knab,
	Vorbis,
	FlatTop3,
	FlatTop5,
]
