export class ColormapControl{
	constructor(selector, defaultSelection){
		this.changed = true;
		this.selector = selector;
		if (defaultSelection === undefined) defaultSelection = 'viridis';
		this.defaultSelection = defaultSelection;
		this.constructor.Colormaps.forEach((cm) => {
			const ele = document.createElement('option');
			ele.value = cm;
			ele.innerHTML = cm;
			selector.appendChild(ele);
			if (defaultSelection == cm) ele.selected = true;
		});
		selector.addEventListener('change', () => {
			this.changed = true;
		});
		window.installThemeChanged(() => {
			this.changed = true;
		});
	}
	addEventListener(e, callback){ this.selector.addEventListener(e, callback); }
	cmap(){
		const cms = this.constructor.Colormaps;
		const find_colormap = this.constructor.find_colormap;
		for (let i = 0; i < cms.length; i++)
			if (this.selector[i].selected)
				return find_colormap(cms[i]);
		return find_colormap(this.defaultSelection);
	}
}

/**
 * Convert HSV to RGB.
 * @param {Number} h - Hue (0-1).
 * @param {Number} s - Saturation (0-1).
 * @param {Number} v - Value (0-1).
 *
 * @returns {String} - rbg color.
 */
export function hsv2rgb(h, s, v){
	let r, g, b;

	const i = Math.floor(h * 6);
	const f = h * 6 - i;
	const p = v * (1 - s);
	const q = v * (1 - f * s);
	const t = v * (1 - (1 - f) * s);

	switch (i % 6){
		case 0:
			r = v;
			g = t;
			b = p;
			break;
		case 1:
			r = q;
			g = v;
			b = p;
			break;
		case 2:
			r = p;
			g = v;
			b = t;
			break;
		case 3:
			r = p;
			g = q;
			b = v;
			break;
		case 4:
			r = t;
			g = p;
			b = v;
			break;
		case 5:
			r = v;
			g = p;
			b = q;
			break;
	}

	return `rgb(${Math.round(r * 255)},${Math.round(g * 255)},${Math.round(b * 255)})`;
}
