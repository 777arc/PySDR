export class SceneTheme{
	constructor(){
		const ttg = document.querySelector('.theme-toggle');
		const _callbacks = [];
		ttg.addEventListener('click', () => {
			if (ttg.innerHTML.includes('dark')) {
				document.documentElement.classList.remove('dark');
				document.documentElement.classList.remove('auto');
				document.documentElement.classList.add('light');
				ttg.innerHTML = 'light';
			}
			else if (ttg.innerHTML.includes('light')) {
				document.documentElement.classList.remove('dark');
				document.documentElement.classList.add('auto');
				document.documentElement.classList.remove('light');
				ttg.innerHTML = 'auto';
			}
			else {
				document.documentElement.classList.add('dark');
				document.documentElement.classList.remove('auto');
				document.documentElement.classList.remove('light');
				ttg.innerHTML = 'dark';
			}
			for (let i = 0; i < _callbacks.length; i++) _callbacks[i](ttg.innerHTML);
		});
		const _install = (cb) => {_callbacks.push(cb);}
		window.installThemeChanged = _install;
		this.installChanged = _install;
	}
}

/**
* Find the URL state provider.
*
* @return {SceneURL}
* */

export function FindSceneURL(){
	if (window.SceneURL === undefined){
		const url = new SceneURL();
		window.SceneURL = url;
		return url;
	}
	return window.SceneURL;
}

export class SceneURL{
	constructor(){
		this.url = new URL(window.location);
		this.url.search = this.url.searchParams;
		this._known = []
		this._needsUpdate = true;
		const _url_checker = () => {
			if (this._needsUpdate){
				history.replaceState({}, null, this.url);
				this._needsUpdate = false;
			}
			window.setTimeout(_url_checker, 200);
		}
		_url_checker();
	}
	/**
	* Set a parameter's value for URL state.
	*
	* @param {String} param Parameter's name
	* @param {any} val Parameter's value
	*
	* @return {null}
	* */
	set_param(param, val){
		const url = this.url;
		url.searchParams.set(param, val);
		url.search = url.searchParams;
		this._needsUpdate = true;
	}
	/**
	* Get a parameter's value from URL state.
	*
	* @param {String} param Parameter's name
	* @param {any} [defaultValue] Parameter's default value (optional)
	*
	* @return {any}
	* */
	get_param(param, defaultValue){
		let res = this.url.searchParams.get(param);
		if ((res === null || res === undefined) && defaultValue !== undefined){
			this.set_param(param, defaultValue);
			res = defaultValue;
		}
		return res;
	}
	/**
	* Remove a parameter from URL state.
	*
	* @param {String} param Parameter's name
	* */
	delete(param){
		this.url.searchParams.delete(param);
		this._needsUpdate = true;
	}
	/**
	* Reset element to default.
	*
	* @param {String} param Parameter/element name
	* @param {HTMLElement} ele HTML DOM element.
	* */
	reset_element(param, ele){
		if (!element_saveable(ele)) return;
		const v = ele.getAttribute('data-default-value');
		if (element_setter(ele, v)) ele.dispatchEvent(new Event('change'));
		this.delete(param);
	}
	/**
	* Bind an element to URL parameter watchlist.
	*
	* @param {String} param Parameter/element name
	* @param {HTMLElement} ele HTML DOM element.
	* @param {Boolean} [forceDispatch=false] Force event dispatch? (default=false)
	* @param {Boolean} [autoUpdate=true] Auto update using element's 'change' event? (default=true)
	* */
	bind_element(param, ele, forceDispatch, autoUpdate){
		if (!element_saveable(ele)) return;
		if (this._known.includes(param)) return;
		this._known.push(param);
		if (autoUpdate === true || autoUpdate === undefined){
			ele.addEventListener('change', () => {
				this.check_element(param, ele);
			});
		}
		ele.setAttribute('__url_bound', true);
		let dv = ele.getAttribute('data-default-value');


		if (dv === null){
			dv = element_getter(ele);
			ele.setAttribute('data-default-value', dv);
		}
		let v = this.get_param(param);
		if (v === undefined || v === null){
			if (forceDispatch) v = dv;
			else return;
		}
		if (element_setter(ele, v) || forceDispatch) ele.dispatchEvent(new Event('change'));
	}
	/**
	* Check if element needs to be added to URL.
	*
	* @param {String} param Parameter/element name
	* @param {HTMLElement} ele HTML DOM element.
	* */
	check_element(param, ele){
		let ev = element_getter(ele);
		if (ev === "" || ele.getAttribute('data-default-value') == ev) this.delete(param);
		else this.set_param(param, ev);
	}
}

export function element_saveable(ele){
	return ['SELECT', 'OPTION', 'INPUT'].includes(ele.nodeName)
}

/**
* Set element's value. Return true if changed.
*
* @param {HTMLElement} ele
* @param {any} value
* @return {Boolean}
* */
export function element_setter(ele, value){
	if (value === undefined || value === null) return false;
	if (ele.type == 'number'){
		if (!isNaN(Number(value))){
			ele.value = value;
			return true;
		}
		return false;
	}
	if (ele.type == 'select-one'){
		for (let i = 0; i < ele.length; i++) {
			const option = ele[i];
			if (option.innerHTML != value) continue;
			option.selected = true;
			return true;
		}
		return false;
	}
	if (ele.type == 'text'){
		ele.value = value;
		return true;
	}
	if (ele.type == 'submit') return false;
	if (ele.type === undefined) return false;
	throw Error(`Unknown element type ${ele.type}.`)
}

/**
* Get element's value.
*
* @param {HTMLElement} ele
* @return {any}
* */
export function element_getter(ele){
	if (ele.type == 'number') return Number(ele.value);
	if (ele.type == 'text') return ele.value;
	if (ele.type == 'submit') return null;
	if (ele.type === undefined) return null;
	if (ele.type == 'select-one'){
		for (let i = 0; i < ele.length; i++) {
			const option = ele[i];
			if (option.selected) return option.innerHTML;
		}
		return ele[0].innerHTML;
	}
	throw Error(`Unknown element type ${ele.type}.`)

}

export class ScenePopup{
	constructor(parent, title, controls, callback){
		this._elements = {};
		const overlay = parent.create_popup_overlay();
		this.parent = parent;
		const ele = document.createElement("div");
		const form = document.createElement("form");
		form.setAttribute('novalidate', 'novalidate');
		ele.classList = "popup";
		let h = document.createElement("h3");
		h.innerHTML = title;
		ele.appendChild(h);
		ele.appendChild(form);
		document.body.appendChild(ele);

		this.container = ele;
		this.overlay = overlay;
		this.form = form;
		this.add_controls(controls);

		const div = document.createElement("div");
		const b1 = document.createElement("button");
		const b2 = document.createElement("button");
		b1.innerHTML = 'OK';
		b2.innerHTML = 'Cancel';
		b2.type = 'button'
		div.classList = 'popup-buttons';
		div.appendChild(b1);
		div.appendChild(b2);
		form.appendChild(div);
		this._focus = null;

		const _hide = () => {
			ele.style.display = 'none';
			overlay.style.display = 'none';
			ele.remove();
		}
		const _notify_cancel = () => {
			_hide();
			if (callback !== undefined) callback(null);
		}
		const _notify_complete = () => {
			_hide();
			if (callback !== undefined) callback(this.build_results());
		}
		overlay.addEventListener('click', _notify_cancel);
		b2.addEventListener('click', _notify_cancel);
		form.addEventListener('submit', (e) => {
			e.preventDefault();
			_notify_complete();
		});
		this.add_action = (text) => {
			const b = document.createElement("button");
			const d = document.createElement("div");
			b.addEventListener('click', _notify_cancel);
			b.type = 'button';
			b.innerHTML = text;
			form.insertBefore(d, div);
			d.appendChild(b);
			d.classList = "popup-buttons";
			return b;
		}
		this.add_note = (text, className) => {
			const d = document.createElement("div");
			d.innerHTML = text;
			form.appendChild(d);
			if (className !== undefined) d.classList = className;
			return d;

		}
	}
	element(key){
		return this._elements[key]['element'];
	}
	set_element_value(key, value){
		const config = this._elements[key];
		const dtype = config['type'];
		if (dtype == "number") config['element'].value = value;
		else if (dtype == "checkbox") config['element'].checked = value;
		else if (dtype == "span") config['element'].innerHTML = value;
	}
	build_results(){
		const results = {};
		for (const [key, entry] of Object.entries(this._elements)){
			let value;
			const etype = entry['type'];
			const ele = entry['element'];
			if (etype == 'number'){
				value = ele.value;
				if ('max' in entry) value = Math.min(entry['max'], value);
				if ('min' in entry) value = Math.max(entry['min'], value);
			}
			else if (etype == 'checkbox') value = ele.checked;
			else if (etype == 'span') value = ele.innerHTML;
			results[key] = value;
		}
		return results;
	}
	add_controls(config){
		if (config === undefined) return;
		config.forEach((e) => {
			const div = document.createElement('div');
			const lbl = document.createElement('label');
			if (!('type' in e)) throw Error("Control config must contain 'type'.");
			if (!('id' in e)) throw Error("Control config must contain 'id'.");
			const etype = e['type'];
			const eid = e['id'];
			const nid = "popup-item-" + eid;
			if (eid in this._elements) throw Error(`Control id '${eid}' is not unique.`);
			let reverse = false;
			let ele;
			if ('label' in e) lbl.innerHTML = e['label'];
			lbl.setAttribute('for', nid);
			if (etype == 'number'){
				ele = document.createElement('input');
				ele.type = 'number';
			}
			else if (etype == 'checkbox'){
				ele = document.createElement('input');
				ele.type = 'checkbox';
			}
			else if (etype == 'span'){
				ele = document.createElement('span');
			}
			else throw Error(`Unknown element type ${etype}`);

			if ('value' in e) ele.value = e['value'];
			if ('min' in e) ele.min = e['min'];
			if ('max' in e) ele.max = e['max'];
			if ('step' in e) ele.step = e['step'];
			ele.id = nid;
			if (reverse){
				lbl.style.textAlign = 'left';
				div.appendChild(ele);
				div.appendChild(lbl);
			}
			else{
				div.appendChild(lbl);
				div.appendChild(ele);
			}
			this.form.appendChild(div);
			e['element'] = ele;
			this._elements[eid] = e;
			if ('focus' in e && e['focus']) this._focus = ele;
		})
	}
	show_from_event(e){
		let ex, ey;
		if (e.type == 'touchstart'){
			ex = e.touches[0].clientX;
			ey = e.touches[0].clientY;
		}
		else{
			ex = e.clientX;
			ey = e.clientY;
		}
		this.show(ex, ey);
	}
	show(cursorX, cursorY){
		const scrollX = window.scrollX;
		const scrollY = window.scrollY;

		const popupWidth = this.container.offsetWidth || 200;
		const popupHeight = this.container.offsetHeight || 100;
		const screenWidth = window.innerWidth + scrollX;
		const screenHeight = window.innerHeight + scrollY;

		let popupX = cursorX + scrollX - popupWidth / 2;
		let popupY = cursorY + scrollY - popupHeight / 2;

		if (popupX < scrollX) popupX = scrollX + 10;
		if (popupY < scrollY) popupY = scrollY + 10;
		if (popupX + popupWidth > screenWidth) popupX = screenWidth - popupWidth - 10;
		if (popupY + popupHeight > screenHeight) popupY = screenHeight - popupHeight - 10;

		this.container.style.left = `${popupX}px`;
		this.container.style.top = `${popupY}px`;
		this.container.style.display = 'flex';
		this.overlay.style.display = 'block';
		if (this._focus !== null){
			this._focus.focus();
			this._focus.select()
		}
	}
}
