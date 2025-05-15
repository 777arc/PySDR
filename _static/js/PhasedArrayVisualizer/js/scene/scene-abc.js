import {MeshColormapControl} from "../cmap/cmap-mesh.js";
import {ListedColormapControl} from "../cmap/cmap-listed.js";
import {SceneQueue} from "./scene-queue.js";
import {ScenePopup, FindSceneURL} from "./scene-util.js";

export class SceneObjectParameterMap{
	/**
	* Create a mapped Parameter for Selecotrs.
	*
	* @param {SceneControlWithSelector} parent
	* @param {String} skey Wrapped key (includes prepend)
	* @param {String} okey Original key (key as it appears in class)
	* @param {Object} cDict Control settings from class
	* */
	constructor(parent, skey, okey, cDict){
		this.parent = parent;
		this.skey = skey;
		this.okey = okey;
		this.ele = parent.find_element(skey);
		this.cDict = cDict;
		this.src = [];
		parent.addEventListener('active-class-changed', (kls) => {
			this.active_class_changed(kls);
		});
		this.ele.addEventListener('change', () => {
			if (this.active) this.last = this.ele.value;
		});
		this.label = document.querySelector("label[for='" + this.ele.id + "']");
		this.div = document.querySelector("#" + this.ele.id + "-div");
		if (!('default' in cDict)) throw Error("Missing 'default'.");
		this.default = cDict['default'];
		this.last = this.default;
		this.title = cDict['title'];
		this.deactivate();
		if (this.div !== null){
			this.hide = () => {this.div.style.display = "none";};
			this.show = () => {this.div.style.display = "flex";};
		}
		else{
			this.hide = () => {};
			this.show = () => {};
		}
	}
	set_value(value){
		this.last = value;
		if (this.active) this.ele.value = value;
	}
	active_class_changed(kls){
		this.set_visible(this.okey in kls.controls);
		this.set_active(this.src.includes(kls));
	}
	add_src(src){ this.src.push(src); }
	set_visible(visible){
		if (visible) this.show();
		else this.hide();
	}
	set_active(active){
		if (active) this.activate();
		else this.deactivate();
	}
	deactivate(){ this.active = false; }
	activate(){
		this.active = true;
		this.ele.setAttribute('data-default-value', this.default);

		['min', 'max', 'step'].forEach((k) => {
			if (k in this.cDict) this.ele.setAttribute(k, this.cDict[k]);
			else this.ele.removeAttribute(k);
		})
		this.ele.value = this.last;

		if (this.label !== null && this.title !== undefined && this.title !== null){
			this.label.innerHTML = this.title;
		}
	}
}

export class SceneObjectABC{
	constructor(prepend, controls, autoUpdateURL){
		if (autoUpdateURL === undefined){
			if (this.constructor.autoUpdateURL !== undefined) autoUpdateURL = this.constructor.autoUpdateURL;
			else autoUpdateURL = true;
		}
		this.autoUpdateURL = autoUpdateURL;
		this.prepend = prepend;
		this.colormap = {};
		this.changed = {};
		this.elements = {};
		this.find_elements(controls);
		this.listeners = {};
		this.controls = controls;
		this.eventTypes = new Set(['control-changed', 'reset']);
		this.queue = null;
		controls.forEach((k) => {
			this.changed[k] = true;
			this.find_element(k).addEventListener('change', () => {
				this.control_changed(k);
			});
		});
		this._children = [];
	}
	/**
	* Install an event listener similar to pure Javascript.
	*
	* Example events:
	*       `control-changed`: (controlname) => {}
	*
	* Classes that inherit this may add their own event types.
	* These can be viewed using calling `list_event_types()`
	* which return all valid event types.
	*
	* @param {String} event
	* @param {function(...):null} callback
	*
	* @return {null}
	* */
	addEventListener(event, callback){
		if (!(this.eventTypes.has(event))){
			throw Error(`'${event} is not a valid event. Expected: ${Array.from(this.eventTypes).join(', ')}`)
		}
		if (!(event in this.listeners)) this.listeners[event] = [];
		this.listeners[event].push(callback);
	}
	async trigger_event(event, ...args){
		if (!(this.eventTypes.has(event))){
			throw Error(`'${event} is not a valid event to trigger.`)
		}
		if (!(event in this.listeners)) return;
		for (const func of this.listeners[event]){ await func(...args); }
	}
	list_event_types(){ return this.eventTypes; }
	add_event_types(...args){ this.eventTypes = this.eventTypes.union(new Set(args)); }

	/**
	* Find DOM element from id. This automatically prepends parent key.
	*
	* @param {String} id Element's ID
	* @param {Boolean} [allowError=true] Throw error if not found? Default=true.
	*
	* @return {HTMLElement}
	* */
	find_element(id, allowError){
		if (this.elements[id] !== undefined) return this.elements[id];
		let eid = this.prepend + "-" + id;
		let ele = document.querySelector("#" + eid);
		if (ele == null && (allowError === undefined || allowError === true)){
			throw Error(`Missing HTML element with id: ${eid}`)
		}
		this.elements[id] = ele;
		return ele;
	}
	find_elements(elements){ elements.forEach((x) => {this.find_element(x)}); }
	create_mesh_colormap_selector(key, defaultSelection){
		const cm = new MeshColormapControl(this.find_element(key), defaultSelection);
		this.colormap[key] = cm;
		return cm;
	}
	create_listed_colormap_selector(key, defaultSelection){
		const cm = new ListedColormapControl(this.find_element(key), defaultSelection);
		this.colormap[key] = cm;
		return cm;
	}
	/**
	* A control with name `key` has changed.
	*
	* @param {String} key
	*
	* @return {null}
	* */
	control_changed(key){
		this.trigger_event('control-changed', key);
		this.changed[key] = true;
	}
	clear_changed(...keys){
		keys.forEach((k) => {
			if (k in this.changed) this.changed[k] = false;
			if (k in this.colormap) this.colormap[k].changed = false;
		});
	}
	create_queue(progressElement, statusElement){
		this.queue = new SceneQueue(progressElement, statusElement);
	}
	create_popup_overlay(){
		let ele = document.querySelector("#popup-overlay");
		if (ele !== undefined && ele !== null) return ele;
		ele = document.createElement("div")
		ele.id = "popup-overlay";
		document.body.appendChild(ele);
		ele.addEventListener('click', () => { ele.style.display = 'none'; });
		return ele;
	}
	create_popup(title, controls, callback){
		return new ScenePopup(this, title, controls, callback);
	}
	add_child(child){ this._children.push(child); }
	all_children(){
		let children = new Set([this]);
		this._children.forEach((c) => {
			children.add(c);
			children = children.union(c.all_children());
		})
		return children;
	}
	children(){ return this._children; }
	reset_all(){ this.trigger_event('reset'); }
}

export class SceneParent extends SceneObjectABC{
	/**
	* A control with name `key` has changed.
	*
	* @param {String} key
	*
	* @return {null}
	* */
	control_changed(key){
		super.control_changed(key);
		this.trigger_event('control-changed', key);
		this.changed[key] = true;
	}
	bind_url_elements(){
		const url = FindSceneURL();
		this._iterate_children_controls((c, k, ele) => {
			url.bind_element(k, ele, false, c.autoUpdateURL);
		});
	}
	update_url_parameters(){
		const url = FindSceneURL();
		this._iterate_children_controls((c, k, ele) => {
			if (c.autoUpdateURL) return;
			url.check_element(k, ele);
		});
	}
	reset_url_parameters(){
		const url = FindSceneURL();
		this._iterate_children_controls((c, k, ele) => {
			url.reset_element(k, ele);
		});
		this.reset_all();
	}
	_iterate_children_controls(caller){
		const cons = new Set([]);
		this.all_children().forEach((c) => {
			for (const [k, cmap] of Object.entries(c.colormap)){
				const ele = cmap.selector;
				if (cons.has(k)) continue;
				caller(c, k, ele);
				cons.add(k);
			}
			for (const [k, ele] of Object.entries(c.elements)){
				if (cons.has(k)) continue;
				caller(c, k, ele);
				cons.add(k);
			}
		});
	}
}

export class SceneControl extends SceneObjectABC{
	/**
	* Build a SceneControl object.
	*
	* @param {SceneParent} parent
	* @param {Array.<String>} controls List of control names required.
	* @param {Boolean} [autoUpdateURL=true] Auto update controls when changing.
	* If false, you must trigger URL update manually using `update_url_parameters`
	*
	* @return {SceneControlTaper}
	* */
	constructor(parent, controls, autoUpdateURL){
		super(parent.prepend, controls, autoUpdateURL);
		this.parent = parent;
		parent.add_child(this);
		parent.addEventListener('reset', () => {this.reset_all()})
	}
	/**
	* Add callable objects to queue.
	*
	* @param {SceneQueue} queue
	*
	* @return {null}
	* */
	add_to_queue(queue){}
}

export class SceneControlWithSelector extends SceneControl{
	constructor(parent, primaryKey, classes, prepend, autoUpdateURL){
		let keys = new Set([primaryKey]);
		classes.forEach((kls) => {
			keys = keys.union(new Set(Object.keys(kls.controls)));
		});
		let wrap_prepend = (vals) => vals;
		let unwrap_prepend = (vals) => vals;
		let wrap_prepend_s = (vals) => vals;
		let unwrap_prepend_s = (vals) => vals;
		if (prepend !== undefined){
			const ks = prepend + "-";
			const ki = ks.length;
			wrap_prepend = (vals) => Array.from(vals, (k) => ks + k);
			unwrap_prepend = (vals) => Array.from(vals, (k) => k.substring(ki));
			wrap_prepend_s = (k) => ks + k;
			unwrap_prepend_s = (k) => k.substring(ki);
			keys = wrap_prepend(keys);
			primaryKey = prepend + "-" + primaryKey;
		}
		super(parent, keys, autoUpdateURL);
		this.add_event_types('primary-changed', 'active-class-changed');
		this.sceneElements = {}

		/** @type {Array<SceneObjectParameterMap>} */
		this.objPars = [];
		this.wrap_prepend = wrap_prepend;
		this.unwrap_prepend = unwrap_prepend;
		this.wrap_prepend_s = wrap_prepend_s;
		this.unwrap_prepend_s = unwrap_prepend_s;
		this.classes = classes;
		this.primarySelector = this.find_element(primaryKey);

		this.mapKey = "__map_" + primaryKey
		classes.forEach((x) => {
			const ele = document.createElement('option');
			ele.value = x.title;
			ele.innerHTML = x.title;
			this.primarySelector.appendChild(ele);
			for (const [k, v] of Object.entries(x.controls)){
				const kk = wrap_prepend_s(k);
				if (kk == primaryKey) continue;
				let smap;
				if (!(this.mapKey in v)){
					v[this.mapKey] = this.objPars.length;
					smap = new SceneObjectParameterMap(this, kk, k, v);
					this.objPars.push(smap);
				}
				else smap = this.objPars[v[this.mapKey]]
				smap.add_src(x);
			}
		});
		this.primarySelector.setAttribute('data-default-value', this.primarySelector[0].innerHTML);
		this.primaryKey = primaryKey;

		const _trigger_change = () => {
			this.trigger_event('primary-changed', primaryKey, this.find_element(primaryKey).value);
			this.trigger_event('active-class-changed', this.selected_class());
		}
		this.primarySelector.addEventListener('change', () => {
			_trigger_change();
		});
		const url = FindSceneURL();
		// bind the primary key first so that it will dispatch change.
		url.bind_element(primaryKey, this.primarySelector, true, this.autoUpdateURL)

		const kls = this.selected_class();
		this.objPars.forEach((obj) => {
			obj.active_class_changed(kls);
		});
	}
	find_object_map(key, kls){
		if (kls === undefined) kls = this.selected_class();
		return this.objPars[kls.controls[key][this.mapKey]]
	}
	selected_class(){
		for (let i = 0; i < this.classes.length; i++){
			if (this.primarySelector[i].selected) return this.classes[i];
		}
		return this.classes[0];
	}
	build_active_object(){
		const kls = this.selected_class();
		let args = [];
		kls.args.forEach((x) => {
			const kk = this.wrap_prepend_s(x);
			const def = kls.controls[x];
			const ele = this.find_element(kk);
			let v = ele.value;
			if (def !== undefined){
				const dtype = def['type'];
				if (dtype == 'float') v = Number(v);
				else if (dtype == 'int') v = parseInt(v);
				else if (dtype === undefined);
				else throw Error(`Unknown data type ${dtype}`);

				const min = def['min'];
				if (min === undefined);
				else if (v < min){
					v = min;
					ele.value = v;
				}
				const max = def['max'];
				if (max === undefined);
				else if (v > max){
					v = max;
					ele.value = v;
				}
			}
			args.push(v);
		})
		return new kls(...args);
	}
	reset_all(){
		for (const obj of Object.values(this.sceneElements)){
			obj.reset();
		}
		super.reset_all();
	}
}

export class SceneControlWithSelectorAutoBuild extends SceneControlWithSelector{
	constructor(parent, primaryKey, classes, htmlElement, prepend){
		let keys = new Set([primaryKey]);

		classes.forEach((kls) => {
			let newKeys = new Set(Object.keys(kls.controls));
			keys = keys.union(newKeys);
		});

		const _k = (k) => {
			let kk = parent.prepend + "-" + k;
			if (prepend === undefined) return kk;
			return parent.prepend + "-" + prepend + "-" + k
		}
		const sel = document.createElement("select");
		sel.style='width:100%';
		sel.id = _k(primaryKey);
		htmlElement.appendChild(sel);

		keys.forEach((k) => {
			if (k === primaryKey) return;
			const name = _k(k);
			const ele = document.createElement('input');
			const div = document.createElement('div');
			const lbl = document.createElement('label');

			div.classList = "form-group";
			div.id = name + "-div";
			ele.type = 'Number';
			ele.id = name;
			ele.name = name;

			lbl.setAttribute('for', name);

			div.appendChild(lbl);
			div.appendChild(ele);
			htmlElement.appendChild(div);
		});

		super(parent, primaryKey, classes, prepend);
	}
}
