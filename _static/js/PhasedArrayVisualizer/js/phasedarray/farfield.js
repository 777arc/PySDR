import {linspace} from "../util.js";

/**
 * @typedef {FarfieldSpherical | FarfieldUV | FarfieldLudwig3} FarfieldHint
 */

export class FarfieldABC{
	static args = ['farfield-ax1-points', 'farfield-ax2-points'];
	static controls = {
		'farfield-ax1-points': {'title': "Theta Points", 'type': "int", 'default': 257, 'min': 1},
		'farfield-ax2-points': {'title': "Phi Points", 'type': "int", 'default': 257, 'min': 1}
	};
	constructor(ax1Points, ax2Points){
		ax1Points = Number(ax1Points)
		ax2Points = Number(ax2Points)
		// ensure samples are odd
		if (ax1Points % 2 == 0) ax1Points++;
		if (ax2Points % 2 == 0) ax2Points++;

		this.farfield_total = new Array(ax2Points);
		this.farfield_log = new Array(ax2Points);
		this.maxValue = -Infinity;
		this.dirMax = null;

		for (let i = 0; i < ax2Points; i++){
			this.farfield_total[i] = new Float32Array(ax1Points);
			this.farfield_log[i] = new Float32Array(ax1Points);
		}
		this.meshPoints = [ax1Points, ax2Points];
	}
	get domain(){ return this.constructor.domain; };
	_yield(text){
		this.ac++;
		return {
			text: text,
			progress: this.ac,
			max: this.maxProgress
		};
	}
	reset_parameters(){
		this.maxValue = -Infinity;
		const [p1, p2] = this.meshPoints;

		this.farfield_im = new Array(p2);
		this.farfield_re = new Array(p2);

		for (let i = 0; i < p2; i++){
			this.farfield_im[i] = new Float32Array(p1);
			this.farfield_re[i] = new Float32Array(p1);
		}
	}
	clear_parameters(){
		const [p1, p2] = this.meshPoints;
		for (let i2 = 0; i2 < p2; i2++){
			for (let i1 = 0; i1 < p1; i1++){
				this.farfield_im[i2][i1] = 0;
				this.farfield_re[i2][i1] = 0;
			}
		}
	}
	calculate_total(total){
		const [p1, p2] = this.meshPoints;
		for (let i2 = 0; i2 < p2; i2++){
			for (let i1 = 0; i1 < p1; i1++){
				const c = Math.abs(this.farfield_re[i2][i1]**2 + this.farfield_im[i2][i1]**2)/total;
				this.farfield_total[i2][i1] = c;
			}
			this.maxValue = Math.max(this.maxValue, ...this.farfield_total[i2]);
		}
		delete this.farfield_im;
		delete this.farfield_re;
	}
	calculate_log(){
		const [p1, p2] = this.meshPoints;
		for (let i2 = 0; i2 < p2; i2++){
			for (let i1 = 0; i1 < p1; i1++){
				this.farfield_log[i2][i1] = 10*Math.log10(this.farfield_total[i2][i1]/this.maxValue);
			}
		}
	}
	create_parameters(pa){
		let ac = 0;
		const maxProgress = pa.geometry.x.length + 4;
		return {
			yield: (text) => {
				ac++;
				return {
					text: text,
					progress: ac,
					max: maxProgress
				}
			},
			x: pa.geometry.x,
			y: pa.geometry.y,
			pha: pa.vectorPhase,
			mag: pa.vectorMag,
		}
	}
	cut(xc, xs, ys, axis){
		xc = Number(xc);
		const mp = Float32Array.from(xs, (x) => Math.abs(x - xc));
		let mv = Infinity;
		let mi = -1;

		for (let i = 0; i < mp.length; i++){
			if (mp[i] < mv){
				mv = mp[i];
				mi = i;
			}
		}
		if (mi < 0) return null;
		if (axis == 0) return ys[mi]
		return Float32Array.from(xs, (_, i) => ys[i][mi])
	}
}

export class FarfieldSpherical extends FarfieldABC{
	static title = 'Spherical';
	static domain = 'spherical';
	constructor(thetaPoints, phiPoints){
		super(thetaPoints, phiPoints);
		[thetaPoints, phiPoints] = this.meshPoints;
		this.thetaPoints = thetaPoints;
		this.phiPoints = phiPoints;

		this.theta = linspace(-Math.PI/2, Math.PI/2, this.thetaPoints);
		this.phi = linspace(-Math.PI/2, Math.PI/2, this.phiPoints);
	}
	*calculator_loop(pa, skipLog){
		const pars = this.create_parameters(pa);
		yield pars.yield('Resetting spherical...');
		this.reset_parameters();
		let sinThetaPi = Float32Array.from({length: this.thetaPoints}, (_, i) => 2*Math.PI*Math.sin(this.theta[i]));
		yield pars.yield('Clearing spherical...');
		this.clear_parameters();
		for (let i = 0; i < pars.x.length; i++){
			yield pars.yield('Calculating spherical re/im...');
			for (let ip = 0; ip < this.phiPoints; ip++){
				const xxv = pars.x[i]*Math.cos(this.phi[ip]);
				const yyv = pars.y[i]*Math.sin(this.phi[ip]);
				for (let it = 0; it < this.thetaPoints; it++){
					const jk = sinThetaPi[it];
					const v = xxv*jk + yyv*jk + pars.pha[i];
					this.farfield_re[ip][it] += pars.mag[i]*Math.cos(v);
					this.farfield_im[ip][it] += pars.mag[i]*Math.sin(v);
				}
			}
		}
		yield pars.yield('Calculating spherical total...');
		this.calculate_total(pars.x.length);
		yield pars.yield('Calculating spherical directivity...');
		this.dirMax = this.compute_directivity();
		if (skipLog === undefined || skipLog === false){
			yield pars.yield('Calculating spherical log...');
			this.calculate_log();
		}
	}
	compute_directivity(){
		let bsa = 0;
		const step = Math.PI/(this.thetaPoints - 1)*Math.PI/(this.phiPoints - 1);
		for (let it = 0; it < this.thetaPoints; it++){
			let st = Math.abs(Math.sin(this.theta[it]))*step;
			for (let ip = 0; ip < this.phiPoints; ip++){
				bsa += this.farfield_total[ip][it]*st;
			}
		}
		return 4*Math.PI*this.maxValue/bsa;
	}
	constant_phi(phi){
		const y = this.cut(Number(phi)*Math.PI/180, this.phi, this.farfield_log, 0);
		if (y === null) return [null, null];
		return [this.theta, y]
	}
	constant_theta(theta){
		const y = this.cut(Number(theta)*Math.PI/180, this.theta, this.farfield_log, 1);
		if (y === null) return [null, null];
		return [this.phi, y]
	}
}

export class FarfieldUV extends FarfieldABC{
	static title = 'UV';
	static domain = 'uv';
	static controls = {
		'farfield-domain': {'title': null},
		'farfield-ax1-points': {'title': "U Points", 'type': "int", 'default': 257, 'min': 1},
		'farfield-ax2-points': {'title': "V Points", 'type': "int", 'default': 257, 'min': 1}
	};
	constructor(uPoints, vPoints, uMax, vMax){
		super(uPoints, vPoints);
		[uPoints, vPoints] = this.meshPoints;
		if (uMax === undefined) uMax = 1;
		if (vMax === undefined) vMax = 1;
		this.uPoints = uPoints;
		this.vPoints = vPoints;
		this.u = linspace(-uMax, uMax, this.uPoints);
		this.v = linspace(-vMax, vMax, this.vPoints);
	}
	*calculator_loop(pa){
		const pars = this.create_parameters(pa);
		yield pars.yield('Resetting UV...');
		this.reset_parameters();
		yield pars.yield('Clearing UV...');
		this.clear_parameters();

		const pi2 = 2*Math.PI;
		for (let i = 0; i < pars.x.length; i++){
			yield pars.yield('Calculating UV re/im...');
			for (let iv = 0; iv < this.vPoints; iv++){
				const xxv = pars.x[i];
				const yyv = pars.y[i]*this.v[iv];
				for (let iu = 0; iu < this.uPoints; iu++){
					const v = (xxv*this.u[iu] + yyv)*pi2 + pars.pha[i];
					this.farfield_re[iv][iu] += pars.mag[i]*Math.cos(v);
					this.farfield_im[iv][iu] += pars.mag[i]*Math.sin(v);
				}
			}
		}
		yield pars.yield('Calculating UV total...');
		this.calculate_total(pars.x.length);

		const sph = new FarfieldSpherical(this.uPoints, this.vPoints);
		const lpi = sph.calculator_loop(pa);
		while (1){
			const n = lpi.next();
			if (n['done']) break;
			yield n['value'];
		}
		this.dirMax = sph.dirMax;
		yield pars.yield('Calculating UV log...');
		this.calculate_log();
	}
	compute_directivity(){ throw Error("Cannot calculate directivity with U-V coordinates."); }
	constant_u(u){
		const y = this.cut(u, this.u, this.farfield_log, 1);
		if (y === null) return [null, null];
		return [this.v, y]
	}
	constant_v(v){
		const y = this.cut(v, this.v, this.farfield_log, 0);
		if (y === null) return [null, null];
		return [this.u, y]
	}
}

export class FarfieldLudwig3 extends FarfieldABC{
	static domain = 'ludwig3';
	static title = 'Ludwig3';
	static controls = {
		'farfield-domain': {'title': null},
		'farfield-ax1-points': {'title': "Az Points", 'type': "int", 'default': 257, 'min': 1},
		'farfield-ax2-points': {'title': "El Points", 'type': "int", 'default': 257, 'min': 1}
	};
	constructor(azPoints, elPoints, azMax, elMax){
		super(azPoints, elPoints);
		[azPoints, elPoints] = this.meshPoints;
		if (azMax === undefined) azMax = 90;
		if (elMax === undefined) elMax = 90;
		this.azPoints = azPoints;
		this.elPoints = elPoints;
		const sc = Math.PI/180
		this.az = linspace(-azMax*sc, azMax*sc, this.azPoints);
		this.el = linspace(-elMax*sc, elMax*sc, this.elPoints);
	}
	*calculator_loop(pa){
		const pars = this.create_parameters(pa);
		yield pars.yield('Resetting Ludwig3...');
		this.reset_parameters();
		yield pars.yield('Clearing Ludwig3...');
		this.clear_parameters();

		const pi2 = 2*Math.PI;
		for (let i = 0; i < pars.x.length; i++){
			yield pars.yield('Calculating Ludwig3 re/im...');
			for (let iv = 0; iv < this.elPoints; iv++){
				const xxv = pars.x[i]*Math.cos(this.el[iv]);
				const yyv = pars.y[i]*Math.sin(this.el[iv]);
				for (let iu = 0; iu < this.azPoints; iu++){
					const w = (xxv*Math.sin(this.az[iu]) + yyv)*pi2 + pars.pha[i];
					this.farfield_re[iv][iu] += pars.mag[i]*Math.cos(w);
					this.farfield_im[iv][iu] += pars.mag[i]*Math.sin(w);
				}
			}
		}
		yield pars.yield('Calculating Ludwig3 total...');
		this.calculate_total(pars.x.length);
		const sph = new FarfieldSpherical(this.azPoints, this.elPoints);
		const lpi = sph.calculator_loop(pa);
		while (1){
			const n = lpi.next();
			if (n['done']) break;
			yield n['value'];
		}
		this.dirMax = sph.dirMax;
		yield pars.yield('Calculating Ludwig3 log...');
		this.calculate_log();
	}
	compute_directivity(){ throw Error("Cannot calculate directivity with Ludwig3 coordinates."); }
	constant_az(az){
		const y = this.cut(az, this.az, this.farfield_log, 1);
		if (y === null) return [null, null];
		return [this.el, y]
	}
	constant_el(el){
		const y = this.cut(el, this.el, this.farfield_log, 0);
		if (y === null) return [null, null];
		return [this.az, y]
	}
}

export const FarfieldDomains = [
	FarfieldSpherical,
	FarfieldUV,
	FarfieldLudwig3,
]
