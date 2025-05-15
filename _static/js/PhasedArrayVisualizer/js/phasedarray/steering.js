import {ludwig3_to_spherical, spherical_to_ludwig3, spherical_to_uv, uv_to_spherical} from "../util.js"

export class SphericalSteeringDeg{
	static title = "Spherical";
	static args = ['theta', 'phi'];
	static controls = {
		'theta': {'title': "Theta (deg)", 'type': "float", 'default': 0},
		'phi': {'title': "Phi (deg)", 'type': "float", 'default': 0}
	};
	constructor(theta, phi){
		this.theta_deg = theta;
		this.phi_deg = phi;
	}
	from(name, v1, v2){
		const n = String(name).toLowerCase();
		if (n == 'spherical') return [v1, v2]
		if (n == 'ludwig3') return ludwig3_to_spherical(v1, v2);
		if (n == 'u-v') return uv_to_spherical(v1, v2);
	}
}

export class Ludwig3SteeringDeg{
	static title = "Ludwig3";
	static args = ['theta', 'phi'];
	static controls = {
		'theta': {'title': "Azimuth (deg)", 'type': "float", 'default': 0, 'step': 0.1},
		'phi': {'title': "Elevation (deg)", 'type': "float", 'default': 0, 'step': 0.1}
	};
	constructor(az, el){
		[this.theta_deg, this.phi_deg] = ludwig3_to_spherical(az, el);
	}
	from(name, v1, v2){
		const n = String(name).toLowerCase();
		if (n == 'ludwig3') return [v1, v2]
		if (n == 'spherical') return spherical_to_ludwig3(v1, v2);
		if (n == 'u-v') {
			[v1, v2] = uv_to_spherical(v1, v2);
			return spherical_to_ludwig3(v1, v2);
		}
	}
}

export class UVSteering{
	static title = "U-V";
	static args = ['theta', 'phi'];
	static controls = {
		'theta': {'title': "U", 'type': "float", 'default': 0, 'step': 0.1},
		'phi': {'title': "V", 'type': "float", 'default': 0, 'step': 0.1}
	};
	constructor(u, v){
		let [th, ph] = uv_to_spherical(u, v);
		if (isNaN(th) || isNaN(ph)){
			th = 0.0;
			ph = 0.0;
		}
		this.theta_deg = th;
		this.phi_deg = ph;
	}
	from(name, v1, v2){
		const n = String(name).toLowerCase();
		if (n == 'u-v') return [v1, v2]
		if (n == 'spherical') return spherical_to_uv(v1, v2);
		if (n == 'ludwig3') {
			[v1, v2] = ludwig3_to_spherical(v1, v2);
			return spherical_to_uv(v1, v2);
		}
	}
}

export const SteeringDomains = [
	SphericalSteeringDeg,
	Ludwig3SteeringDeg,
	UVSteering,
]
