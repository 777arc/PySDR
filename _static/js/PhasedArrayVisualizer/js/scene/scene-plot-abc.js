import {SceneObjectABC} from "./scene-abc.js"
/** @import { ColormapControl } from "../cmap/cmap-util.js" */
/** @import { SceneParent } from "./scene-abc.js" */

export class ScenePlotABC extends SceneObjectABC{
	/**
	* Create a new plot on a canvas.
	*
	* @param {SceneParent} parent
	* @param {HTMLCanvasElement} canvas
	* @param {ColormapControl} cmap
	* */
	constructor(parent, canvas, cmap){
		super(parent.prepend, []);
		parent.add_child(this);
		this.parent = parent;
		this.canvas = canvas;
		this.cmap = cmap;
	}
	install_scale_control(key){
		const ele = this.find_element(key);
		const _val = () => {
			const v = -Math.max(5, Math.abs(ele.value));
			ele.value = Math.abs(v);
			return v;
		}
		ele.addEventListener('change', () => {
			this.min = _val();
			this.trigger_event('data-min-changed', this.min);
		})
		this.min = _val();
	}
	create_hover_items(){
		const canvas = this.canvas;
		const p = canvas.parentElement.parentElement;
		const h = p.querySelector(".canvas-header");
		const ele = document.createElement("div");
		ele.classList = "canvas-hover-div";
		ele.innerHTML = "&nbsp;";
		canvas.hover_container = ele;
		h.appendChild(ele);

		canvas.addEventListener('mouseleave', () => {
			if (this.queue !== undefined && this.queue.running) return;
			canvas.hover_container.innerHTML = "&nbsp";
		});
	}
	create_progress_bar(){
		const canvas = this.canvas;
		const p = canvas.parentElement.parentElement;
		const h = p.querySelector(".canvas-header");
		const ele = document.createElement("progress");
		ele.value = 100;
		ele.max = 100;
		h.appendChild(ele);
	}
	create_queue(){
		const canvas = this.canvas;
		const p = canvas.parentElement.parentElement;
		const h = p.querySelector(".canvas-header");
		super.create_queue(h.querySelector("progress"), h.querySelector(".canvas-hover-div"));
	}
}
