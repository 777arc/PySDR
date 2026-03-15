/**
 * Queue iterator object.
 *
 * @typedef {Object} QueueIteratorResult
 * @property {String} text - Text to display on progress bar.
 * @property {number} progress - Progress value.
 * @property {number} max - Maximum value expected on `progress`.
 */

export class SceneQueue{
	constructor(progressElement, statusElement){
		this.progress = progressElement;
		this.status = statusElement;
		this.reset();
		this.channel = new MessageChannel();
		this.channel.port1.onmessage = () => {this.process_queue()};
		this.running = false;
	}
	/**
	* Add callable object to queue.
	*
	* @param {String} text String to display on status bar
	* @param {function():null} func Callback function.
	*
	* @return {null}
	* */
	add(text, func){
		this.queue.push({
			text: text,
			func: func,
			type: 'function',
		});
	}
	/**
	* Add iterator to queue. Progress bar will be updated to match
	* progress of iterator.
	*
	* @param {String} text String to display on status bar
	* @param {function():Iterator<QueueIteratorResult>} func Callback iterator that yields information.
	*
	* @return {null}
	* */
	add_iterator(text, func){
		this.queue.push({
			text: text,
			func: func,
			type: 'iterator',
		});
	}
	dump(){
		this.queue.forEach((entry) => {
			console.log("---- " + entry['text']);
			console.log(entry['func']);
		});
	}
	next(){
		if (this.queue.length == 0) return null;
		return this.queue.shift();
	}
	reset(){
		this.queue = [];
		this.startingLength = 0;
		this._current = null;
	}
	start(finalText){
		if (finalText === undefined) finalText = "Complete";
		this.finalText = finalText;
		const prog = this.progress;
		this.startingLength = this.queue.length;
		this._current = null;
		prog.value = 0;
		prog.max = this.startingLength;
		if (!this.running) this.process_queue();
	}
	get length(){ return this.queue.length; }

	log(string, toConsole){
		if (string !== undefined){
			if (toConsole !== false) console.log(string)
			this.status.innerHTML = string;
		}
	}
	process_queue(){
		const prog = this.progress;
		this.running = true;
		if (this._current === null){
			this._current = this.next();
			prog.value = prog.max - this.length;
			if (this._current === null){
				this.log(this.finalText, false);
				this.running = false;
				return;
			}
			this.log(this._current['text']);
		}
		else{
			let c = this._current;
			if (c['type'] == 'next'){
				let v = c['func'].next();
				if (v.done) {
					this._current = null;
					prog.max = this.startingLength;
					prog.value = prog.max - this.length;
				}
				else{
					this.log(v.value['text']);
					prog.max = v.value['max'];
					prog.value = v.value['progress'];
				}
			}
			else if (c['type'] == 'iterator'){
				this._current['type'] = 'next';
				this._current['func'] = c['func']();
			}
			else {
				c['func']();
				this._current = null;
			}
		}
		this.channel.port2.postMessage("");
	}
}
