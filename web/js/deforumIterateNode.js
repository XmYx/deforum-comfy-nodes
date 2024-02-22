import { app } from "../../../scripts/app.js";
import { api } from '../../../scripts/api.js'
import { ComfyWidgets } from "../../../scripts/widgets.js";

// Many functions copied from:
// https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite
// Thank you!


document.getElementById("comfy-file-input").accept += ",video/webm,video/mp4";

function chainCallback(object, property, callback) {
    if (object == undefined) {
        //This should not happen.
        console.error("Tried to add callback to non-existant object")
        return;
    }
    if (property in object) {
        const callback_orig = object[property]
        object[property] = function () {
            const r = callback_orig.apply(this, arguments);
            callback.apply(this, arguments);
            return r
        };
    } else {
        object[property] = callback;
    }
}

async function uploadFile(file) {
    //TODO: Add uploaded file to cache with Cache.put()?
    try {
        // Wrap file in formdata so it includes filename
        const body = new FormData();
        const i = file.webkitRelativePath.lastIndexOf('/');
        const subfolder = file.webkitRelativePath.slice(0,i+1)
        const new_file = new File([file], file.name, {
            type: file.type,
            lastModified: file.lastModified,
        });
        body.append("image", new_file);
        if (i > 0) {
            body.append("subfolder", subfolder);
        }
        const resp = await api.fetchApi("/upload/image", {
            method: "POST",
            body,
        });

        if (resp.status === 200) {
            return resp.status
        } else {
            alert(resp.status + " - " + resp.statusText);
        }
    } catch (error) {
        alert(error);
    }
}

function addUploadWidget(nodeType, nodeData, widgetName, type="video") {
    chainCallback(nodeType.prototype, "onNodeCreated", function() {
        const pathWidget = this.widgets.find((w) => w.name === widgetName);
        const fileInput = document.createElement("input");
//        chainCallback(this, "onRemoved", () => {
//            fileInput?.remove();
//        });
        if (type == "folder") {
            Object.assign(fileInput, {
                type: "file",
                style: "display: none",
                webkitdirectory: true,
                onchange: async () => {
                    const directory = fileInput.files[0].webkitRelativePath;
                    const i = directory.lastIndexOf('/');
                    if (i <= 0) {
                        throw "No directory found";
                    }
                    const path = directory.slice(0,directory.lastIndexOf('/'))
                    if (pathWidget.options.values.includes(path)) {
                        alert("A folder of the same name already exists");
                        return;
                    }
                    let successes = 0;
                    for(const file of fileInput.files) {
                        if (await uploadFile(file) == 200) {
                            successes++;
                        } else {
                            //Upload failed, but some prior uploads may have succeeded
                            //Stop future uploads to prevent cascading failures
                            //and only add to list if an upload has succeeded
                            if (successes > 0) {
                                break
                            } else {
                                return;
                            }
                        }
                    }
                    pathWidget.options.values.push(path);
                    pathWidget.value = path;
                    if (pathWidget.callback) {
                        pathWidget.callback(path)
                    }
                },
            });
        } else if (type == "video") {
            Object.assign(fileInput, {
                type: "file",
                accept: "video/webm,video/mp4,video/mkv,image/gif",
                style: "display: none",
                onchange: async () => {
                    if (fileInput.files.length) {
                        if (await uploadFile(fileInput.files[0]) != 200) {
                            //upload failed and file can not be added to options
                            return;
                        }
                        const filename = fileInput.files[0].name;
                        pathWidget.options.values.push(filename);
                        pathWidget.value = filename;
                        if (pathWidget.callback) {
                            pathWidget.callback(filename)
                        }
                    }
                },
            });
        } else {
            throw "Unknown upload type"
        }
        document.body.append(fileInput);
        let uploadWidget = this.addWidget("button", "choose " + type + " to upload", "image", () => {
            //clear the active click event
            app.canvas.node_widget = null

            fileInput.click();
        });
        uploadWidget.options.serialize = false;
    });
}


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
	beforeRegisterNodeDef(nodeType, nodeData) {
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
                const v = app.nodeOutputs?.[this.id + ""];
                if (!this.flags.collapsed && v) {

                    const counter = v["counter"]
                    const max_frames = v["max_frames"]

                    console.log("COUNTER")
                    console.log(counter[0])
                    console.log("MAX FRAMES")
                    console.log(max_frames[0])

                    if (counter[0] >= max_frames[0]) {
                        //document.getElementById('autoQueueCheckbox').checked = false;
                        if (document.getElementById('autoQueueCheckbox').checked === true) {
                            document.getElementById('autoQueueCheckbox').click();
                        }
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
		} else if (nodeType.comfyClass === "DeforumLoadVideo") {
                console.log("FOUND")
                addUploadWidget(nodeType, nodeData, "video");

		};
	},
});
