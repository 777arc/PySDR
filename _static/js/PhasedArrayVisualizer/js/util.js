export const c_deg2rad = Math.PI/180;
export const c_rad2deg = 180/Math.PI;

/**
 * Convert input radians to degrees.
 *
 * @param {Float32Array} radians
 *
 * @return {Float32Array}
 * */
export function rad2deg(radians){
	return Float32Array.from(radians, r => r*c_rad2deg);
}

/**
 * Convert input degrees to radians.
 *
 * @param {Float32Array} degrees
 *
 * @return {Float32Array}
 * */
export function deg2rad(degrees){
	return Float32Array.from(degrees, r => r*c_deg2rad);
}

/**
 * Create a uniformly spaced Float32Array with `num` steps.
 *
 * @param {Number} start
 * @param {Number} stop
 * @param {Number | int} num
 *
 * @return {Float32Array}
 * */
export function linspace(start, stop, num){
	return Float32Array.from({length: num}, (_, i) => start + (stop - start)/(num - 1) * i);
}

/**
 * Create a range Float32Array.
 *
 * @param {Number | int} [start=0] (optional)
 * @param {Number | int} stop
 * @param {Number | int} [step=1] (optional)
 *
 * @return {Float32Array}
 * */
export function arange(start, stop, step){
	if (stop === undefined){
		stop = start;
		start = 0;
	}
	if (step === undefined) step = 1;
	else step = parseInt(step);
	let a = []
	for (let i=start; i < stop; i += step) a.push(i);
	return Float32Array.from(a);
}

/**
* Calculate factorial of a number in log form.
*
* @param {Number} v
*
* @return {Number}
* */
export function factorial_log(v){
	if (v == 0) return 0.0;
	if (v < 0) throw Error("Factorial of a negative number is unknown.");
	const a = arange(1, parseInt(value) + 1);
	let s = 0;
	for (let i = 0; i < a.length; i++) s += Math.log10(a[i]);
	return s;
}

/**
* Calculate factorial of a number.
*
* @param {Number} v
*
* @return {Number}
* */
function factorial(v) {
	if (v === 0) return 1;
	let s = 1.0;
	for (let i = 1; i <= v; i++) s *= i;
	return s;
}

/**
* Normalizes input to be between [-m, m].
*
* @param {Float32Array} x
* @param {Number} [m=0.5] (optional) Normalization bounds (default=0.5).
* @param {Number} [c=0.0] (optional) Center (default=0.0).
*
* @return {Float32Array}
* */
export function normalize(x, m, c){
	if (m === undefined) m = 0.5;
	if (c === undefined) c = 0.0;
	const maxX = Math.max(...x);
	const minX = Math.min(...x);
	const den = (maxX - minX)/(m*2);
	return Float32Array.from({'length': x.length}, (_, i) => (x[i] - minX)/den - m + c);
}
/**
* Gamma function.
*
* @param {Number} z
*
* @return {Number}
* */
export function gamma(z){
	const g = 7;
	const p = [
		0.99999999999980993,
		676.5203681218851,
		-1259.1392167224028,
		771.32342877765313,
		-176.61502916214059,
		12.507343278686905,
		-0.13857109526572012,
		9.9843695780195716e-6,
		1.5056327351493116e-7,
	];
	if (z < 0.5) return Math.PI / (Math.sin(Math.PI*z)*gamma(1 - z));
	else {
		z -= 1;
		let x = p[0];
		for (let i = 1; i < g + 2; i++) x += p[i] / (z + i);
		const t = z + g + 0.5;
		return Math.sqrt(2*Math.PI)*Math.pow(t, z + 0.5)*Math.exp(-t)*x;
	}
  }

/**
* Create an array of ones.
*
* @param {Number | int} len
*
* @return {Float32Array}
* */
export function ones(len){ return Float32Array.from({length: len}, () => 1); }

/**
* Create an array of zeros.
*
* @param {Number | int} len
*
* @return {Float32Array}
* */
export function zeros(len){ return Float32Array.from({length: len}, () => 0); }

/**
* Calculate the zero-order Modified Bessel function.
*
* @param {Number} x
* @param {Number | int} [maxIter=60] (optional) Max iteration
* @param {Number} [tolerance=1e-9] (optional) Calculation tolerance
*
* @return {Number}
* */
export function bessel_modified_0(x, maxIter, tolerance){
	if (maxIter === undefined) maxIter = 50;
	if (tolerance === undefined) tolerance = 1e-9;
	let s = 0;
	for (let i = 0; i <= maxIter; i++){
		let t = (1/(factorial(i))*(x/2)**i)**2;
		s += t;
		if (Math.abs(t) <= tolerance) break;
	}
	return 1 + s;
}

/**
* Adjust theta/phi such that each are [-90, 90] (or [-PI/2, PI/2]).
*
* @param {Number} theta
* @param {Number} phi
* @param {Boolean} [deg=true] (optional) Input/Output in deg?
*
* @return {[Number, Number]} theta, phi
* */
export function adjust_theta_phi(theta, phi, deg){
	let o1 = 180;
	if (deg !== undefined && !deg) o1 = Math.PI;
	let o2 = o1/2.0;
	if (phi > o2){
		phi -= o1;
		theta = -theta;
	}
	if (phi < -o2){
		phi += o1;
		theta = -theta;
	}
	return [theta, phi]
}

/**
* Convolve two arrays.
*
* @param {Float32Array} array1
* @param {Float32Array} array2
*
* @return {Float32Array}
* */
export function convolve(array1, array2){
	const len1 = array1.length;
	const len2 = array2.length;
	return Float32Array.from({length: len1 + len2 - 1}, (_, i) => {
		let sum = 0;
		for (let j = 0; j < len2; j++){
			if (i - j >= 0 && i - j < len1) sum += array1[i - j] * array2[j];
	  	}
		return sum;
	})
}

/**
* Convert az/el (Ludwig-3) to theta/phi (Spherical).
*
* https://www.mathworks.com/help/phased/ref/azel2phitheta.html
*
* @param {Number} az (deg)
* @param {Number} el (deg)
*
* @return {[Number, Number]} theta, phi (deg)
* */
export function ludwig3_to_spherical(az, el){
	const raz = c_deg2rad*az;
	const rel = c_deg2rad*el;
	return [
		Math.acos(Math.cos(rel)*Math.cos(raz))*c_rad2deg,
		Math.atan2(Math.tan(rel), Math.sin(raz))*c_rad2deg
	];
}

/**
* Convert theta/phi (Spherical) to az/el (Ludwig-3).
*
* https://www.mathworks.com/help/phased/ref/azel2phitheta.html
*
* @param {Number} theta (deg)
* @param {Number} phi (deg)
*
* @return {[Number, Number]} az, el (deg)
* */
export function spherical_to_ludwig3(theta, phi){
	const rth = c_deg2rad*theta;
	const rph = c_deg2rad*phi;
	return [
		Math.atan(Math.cos(rph)*Math.tan(rth))*c_rad2deg,
		Math.asin(Math.sin(rph)*Math.sin(rth))*c_rad2deg
	];
}

/**
* Convert theta/phi (Spherical) to u/v.
*
* https://www.mathworks.com/help/phased/ref/phitheta2uv.html
*
* @param {Number} theta (deg)
* @param {Number} phi (deg)
*
* @return {[Number, Number]} u, v
* */
export function spherical_to_uv(theta, phi){
	const rth = c_deg2rad*theta;
	const rph = c_deg2rad*phi;
	return [
		Math.sin(rth)*Math.cos(rph),
		Math.sin(rth)*Math.sin(rph)
	];
}

/**
* Convert u/v to theta/phi (Spherical).
*
* https://www.mathworks.com/help/phased/ref/phitheta2uv.html
*
* @param {Number} u (deg)
* @param {Number} v (deg)
* @param {Boolean} [adjust=true] Adjust U/V to ensure |U| < 1 && |V| < 1
*
* @return {[Number, Number]} theta, phi (deg)
* */
export function uv_to_spherical(u, v, adjust){
	if (adjust === true || adjust === undefined){
		while (u > 1.0) u -= 1.0;
		while (u < -1.0) u += 1.0;
		while (v > 1.0) v -= 1.0;
		while (v < -1.0) v += 1.0;
	}
	if (Math.sqrt(u**2 + v**2) > 1) return [NaN, NaN]
	return [
		Math.asin(Math.sqrt(u**2 + v**2))*c_rad2deg,
		Math.atan2(v, u)*c_rad2deg
	];
}
