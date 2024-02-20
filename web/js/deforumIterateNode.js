import { app } from "../../../scripts/app.js";
import { ComfyWidgets } from "../../../scripts/widgets.js";

app.registerExtension({
	name: "deforum.deforumIterator",
	init() {
		const STRING = ComfyWidgets.STRING;
		ComfyWidgets.STRING = function (node, inputName, inputData) {
			const r = STRING.apply(this, arguments);
			r.widget.dynamicPrompts = inputData?.[1].dynamicPrompts;
			return r;
		};
	},
	beforeRegisterNodeDef(nodeType) {
		if (nodeType.comfyClass === "DeforumIteratorNode") {
            const onExecuted = nodeType.prototype.onExecuted
            nodeType.prototype.onExecuted = function (message) {

            const r = onExecuted ? onExecuted.apply(this, message) : undefined
                for (const w of this.widgets || []) {
                    if (w.name === "reset_counter") {
                        const counterWidget = w;
                        counterWidget.value = false;
                    }
                }
            return r
            }

            const onDrawForeground = nodeType.prototype.onDrawForeground;
			nodeType.prototype.onDrawForeground = function (ctx) {
				const r = onDrawForeground?.apply?.(this, arguments);
				const v = app.nodeOutputs?.[this.id + ""];
				if (!this.flags.collapsed && v) {

					const text = v["counter"] + "";
					ctx.save();
					ctx.font = "bold 48px sans-serif";
					ctx.fillStyle = "dodgerblue";
					const sz = ctx.measureText(text);
					ctx.fillText(text, (this.size[0]) / 2 - sz.width - 5, LiteGraph.NODE_SLOT_HEIGHT * 3);
					ctx.restore();
				}

				return r;
			};
		}
	},
});
