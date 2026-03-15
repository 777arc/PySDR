import {ColormapControl} from "./cmap-util.js"

export class ListedColormapControl extends ColormapControl{
	static Colormaps = [
		'Vibrant',
		'Bright',
		'Muted',
		'Vibrant_r',
		'Bright_r',
		'Muted_r',
	]
	static find_colormap = find_colormap;
}
/**
 * @callback Colormapper
 * @param {number} val - Value for colormap. Will be bound [0, 1].
 *
 * @return {string} Color as `rgb()` string.
 */

/**
 * Find and return colormap based on input `name`.
 *
 * @param {string} name
 *
 * @return {Colormapper}
 * */
export function find_colormap(name){
	const rev = name.endsWith('_r');
	switch (name){
		case 'Bright':
		case 'Bright_r':
			return _array_wrapper(_tol_bright, rev);

		case 'Vibrant':
		case 'Vibrant_r':
			return _array_wrapper(_tol_vibrant, rev);

		case 'Muted':
		case 'Muted_r':
			return _array_wrapper(_tol_muted, rev);

		default:
			throw Error(`Missing colormap: ${name}.`)
	}
}

function _array_wrapper(colors, reverse){
	if (reverse === undefined) reverse = false;
	return (value) => {
		if (isNaN(value)) return `rgb(0,0,0)`
		if (!isFinite(value)) return `rgb(0,0,0)`
		if (reverse) value = colors.length - value - 1;
		return colors[value % colors.length];
	}
}

const _tol_bright = [
	"#4477AA",
	"#EE6677",
	"#228833",
	"#CCBB44",
	"#66CCEE",
	"#AA3377",
	"#BBBBBB",
]

const _tol_vibrant = [
	"#EE7733",
	"#0077BB",
	"#33BBEE",
	"#EE3377",
	"#CC3311",
	"#009988",
	"#BBBBBB",
]

const _tol_muted = [
	"#CC6677",
	"#332288",
	"#DDCC77",
	"#117733",
	"#88CCEE",
	"#882255",
	"#44AA99",
	"#999933",
	"#AA4499",
]
