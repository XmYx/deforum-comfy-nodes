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
function fitHeight(node) {
    node.setSize([node.size[0], node.computeSize([node.size[0], node.size[1]])[1] + 20])
    node?.graph?.setDirtyCanvas(true);
}
function addVideoPreview(nodeType) {
    chainCallback(nodeType.prototype, "onNodeCreated", function() {
        var element = document.createElement("div");
        const previewNode = this;
        previewNode.size[1] += 45;
        var previewWidget = this.addDOMWidget("videopreview", "preview", element, {
            serialize: false,
            hideOnZoom: false,
            getValue() {
                return element.value;
            },
            setValue(v) {
                element.value = v;
            },
        });
        previewWidget.computeSize = function(width) {
            if (this.aspectRatio && !this.parentEl.hidden) {
                let height = (previewNode.size[0]-20)/ this.aspectRatio + 10;
                if (!(height > 0)) {
                    height = 0;
                }
                this.computedHeight = height + 10;
                return [width, height];
            }
            return [width, -4];//no loaded src, widget should not display
        }
        //element.style['pointer-events'] = "none"
        previewWidget.value = {hidden: false, paused: false, params: {}}
        previewWidget.parentEl = document.createElement("div");
        previewWidget.parentEl.className = "deforumVideoSavePreview";
        previewWidget.parentEl.style['width'] = "100%"
        element.appendChild(previewWidget.parentEl);
        previewWidget.imgEl = document.createElement("img");
        previewWidget.imgEl.style['width'] = "100%"
        previewWidget.imgEl.hidden = true;


        // Add transport controls
        const controls = document.createElement("div");
        controls.style.display = "flex";
        controls.style.justifyContent = "space-around";
        previewWidget.parentEl.appendChild(controls);

        // Play button
        const playButton = document.createElement("button");
        playButton.innerText = "Play";
        playButton.onclick = () => {
            if (!this.playing) {
                this.startPlayback(this.playbackInterval);
            }
        };
        controls.appendChild(playButton);

        // Stop button
        const stopButton = document.createElement("button");
        stopButton.innerText = "Stop";
        stopButton.onclick = () => {
            this.stopPlayback();
            // Reset frameIndex if needed
            // frameIndex = 0; // Uncomment to reset frame index on stop
        };
        controls.appendChild(stopButton);

        // Step Back button
        const stepBackButton = document.createElement("button");
        stepBackButton.innerText = "Step Back";
        stepBackButton.onclick = () => {
            if (frameIndex > 0) {
                frameIndex -= 1;
            } else {
                frameIndex = cachedFrames.length - 1; // Wrap around to the last frame
            }
            updateFrame(frameIndex);
        };
        controls.appendChild(stepBackButton);

        // Step Forward button
        const stepForwardButton = document.createElement("button");
        stepForwardButton.innerText = "Step Forward";
        stepForwardButton.onclick = () => {
            frameIndex = (frameIndex + 1) % cachedFrames.length;
            updateFrame(frameIndex);
        };
        controls.appendChild(stepForwardButton);


        previewWidget.parentEl.appendChild(previewWidget.imgEl)
        let frameIndex = 0;
        let cachedFrames = this.getCachedFrames(); // Assuming this method exists and retrieves an array of frame data
        previewWidget.imgEl.onload = () => {
            previewWidget.aspectRatio = previewWidget.imgEl.naturalWidth / previewWidget.imgEl.naturalHeight;
            fitHeight(this);
        };
        // Function to update frame view
        function updateFrame(index) {
            cachedFrames = previewNode  .getCachedFrames()
            if (cachedFrames && cachedFrames.length > 0) {
                previewWidget.imgEl.hidden = false;
                previewWidget.imgEl.src = 'data:image/png;base64,' + cachedFrames[index];
            }
        }


        this.playing = false;
        this.playbackInterval = 80;
        this.startPlayback = function(playbackInterval) {
            if (this.playing) {
                this.stopPlayback(); // Stop current playback if it's running
            }

            this.playbackInterval = playbackInterval;
            const widget = this; // Capture 'this' to use inside setInterval function
            this.imageSequenceInterval = setInterval(() => {
                const cachedFrames = this.getCachedFrames();
                //const displayFrames = cachedFrames.length > 0 ? cachedFrames : frames;
                if (cachedFrames && cachedFrames.length > 0) {
                    previewWidget.imgEl.hidden = false;
                    previewWidget.imgEl.src = 'data:image/png;base64,' + cachedFrames[frameIndex];
                    frameIndex = (frameIndex + 1) % cachedFrames.length;
                }
            }, this.playbackInterval); // Update frame every 80ms
        };
        // Function to stop playback
        this.stopPlayback = function() {
            if (this.imageSequenceInterval) {
                clearInterval(this.imageSequenceInterval);
                this.imageSequenceInterval = null; // Clear the interval ID
                this.playing = false; // Mark as not playing
            }
        };
        this.setPlaybackInterval = function(newInterval) {
            this.playbackInterval = newInterval;
            if (this.playing) {
                this.stopPlayback();
                this.startPlayback(newInterval); // Restart playback with new interval if it's currently playing
            }
        };

    });
}



function addUploadWidget(nodeType, nodeData, widgetName, type="video") {
    chainCallback(nodeType.prototype, "onNodeCreated", function() {
        const pathWidget = this.widgets.find((w) => w.name === widgetName);
        const fileInput = document.createElement("input");
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

// Extend the node prototype to include frame caching capabilities
function extendNodePrototypeWithFrameCaching(nodeType) {
    nodeType.prototype.frameCache = []; // Initialize an empty cache

    // Method to add frames to the cache
    nodeType.prototype.cacheFrames = function(frames) {
        this.frameCache = this.frameCache.concat(frames);
    };

    // Method to clear the frame cache
    nodeType.prototype.clearFrameCache = function() {
        this.frameCache = [];
    };

    // Method to get cached frames
    nodeType.prototype.getCachedFrames = function() {
        return this.frameCache;
    };
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
            const onIteratorExecuted = nodeType.prototype.onExecuted
            nodeType.prototype.onExecuted = function (message) {
                const r = onIteratorExecuted ? onIteratorExecuted.apply(this, message) : undefined
                for (const w of this.widgets || []) {
                    if (w.name === "reset_counter") {
                        const counterWidget = w;
                        counterWidget.value = false;
                    } else if (w.name === "reset_latent") {
                        const resetWidget = w;
                        resetWidget.value = false;
                    }
                }
                const v = app.nodeOutputs?.[this.id + ""];
                if (!this.flags.collapsed && v) {

                    const counter = v["counter"]
                    const max_frames = v["max_frames"]
                    const enableAutorun = v["enable_autoqueue"][0]



                    if (counter[0] >= max_frames[0]) {
                        if (document.getElementById('autoQueueCheckbox').checked === true) {
                            document.getElementById('autoQueueCheckbox').click();
                        }
                    }

                    if (enableAutorun === true) {
                        if (document.getElementById('autoQueueCheckbox').checked === false) {
                            document.getElementById('autoQueueCheckbox').click();
                            document.getElementById('extraOptions').style.display = 'block';
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
                addUploadWidget(nodeType, nodeData, "video");

		} else if (nodeType.comfyClass === "DeforumVideoSaveNode") {
            extendNodePrototypeWithFrameCaching(nodeType);
            addVideoPreview(nodeType);
            const onVideoSaveExecuted = nodeType.prototype.onExecuted
            nodeType.prototype.onExecuted = function (message) {

            const r = onVideoSaveExecuted ? onVideoSaveExecuted.apply(this, message) : undefined

                let swapSkipSave = false;
                for (const w of this.widgets || []) {
                    if (w.name === "dump_now") {
                        const dumpWidget = w;
                        if (dumpWidget.value === true) {
                            swapSkipSave = true
                        }
                        dumpWidget.value = false;
                        this.shouldResetAnimation = true;
                    } else if (w.name === "skip_save") {
                        const saveWidget = w;
                        if (swapSkipSave === true) {
                            saveWidget.value = false;
                        }


                    }
                }
                const output = app.nodeOutputs?.[this.id + ""];
                const should_reset = output["should_dump"]
                const fps = output["fps"]
                const millisecondsPerFrame = 1000 / fps[0];



                if (output && "frames" in output) { // Safely check if 'frames' is a key in 'output'
                    if (this.playing === false) {
                        this.playing = true;
                        this.cacheFrames(output["frames"]);
                        this.startPlayback(millisecondsPerFrame);
                    } else {
                        this.setPlaybackInterval(millisecondsPerFrame);
                        this.cacheFrames(output["frames"]);
                    }

                    if (should_reset[0] === true) {
                        this.stopPlayback();
                        this.clearFrameCache();
                        this.cacheFrames(output["frames"]);
                    }
                }


            return r
            }
            const onVideoSaveForeground = nodeType.prototype.onDrawForeground;
            nodeType.prototype.onDrawForeground = function (ctx) {
                const r = onVideoSaveForeground?.apply?.(this, arguments);
                const v = app.nodeOutputs?.[this.id + ""];
                if (!this.flags.collapsed && v) {

                    const text = v["counter"] + " cached frame(s)";
                    ctx.save();

                    // Set font for measuring text width and for drawing
                    ctx.font = "bold 14px sans-serif";

                    // Measure text to center it on the rectangle
                    const sz = ctx.measureText(text);
                    const textWidth = sz.width;
                    const textHeight = 14; // Approximation based on font size

                    // Rectangle dimensions and position
                    const rectWidth = textWidth + 20; // Padding around text
                    const rectHeight = textHeight + 10; // Padding around text
                    const rectX = (this.size[0] - rectWidth) / 2;
                    const rectY = LiteGraph.NODE_TITLE_HEIGHT - rectHeight / 2 - 15;

                    // Draw rectangle
                    ctx.fillStyle = "rgba(0, 0, 0, 0.8)"; // Semi-transparent dark rectangle
                    ctx.fillRect(rectX, rectY, rectWidth, rectHeight);

                    // Draw text centered in the rectangle
                    ctx.fillStyle = "white"; // White text color
                    const textX = (this.size[0] - textWidth) / 2;
                    const textY = LiteGraph.NODE_TITLE_HEIGHT + textHeight / 2 - 15; // Adjust based on the font size

                    ctx.fillText(text, textX, textY);
                    ctx.restore();


                }

                return r;
            };

		};
	},
});
class FloatingConsole {
    constructor() {
        this.element = document.createElement('div');
        this.element.id = 'floating-console';
        this.titleBar = this.createTitleBar();
        this.contentContainer = this.createContentContainer();

        this.element.appendChild(this.titleBar);
        this.element.appendChild(this.contentContainer);

        document.body.appendChild(this.element);

        this.dragging = false;
        this.prevX = 0;
        this.prevY = 0;

        this.setupStyles();
        this.addEventListeners();
        this.addMenuButton();
    }

    setupStyles() {
        Object.assign(this.element.style, {
            position: 'fixed',
            bottom: '10px',
            right: '10px',
            width: '300px',
            maxHeight: '600px',
            backgroundColor: 'rgba(0, 0, 0, 0.8)',
            color: 'white',
            borderRadius: '5px',
            zIndex: '1000',
            display: 'none', // Consider starting visible for debugging
            boxSizing: 'border-box',
            overflow: 'hidden',
            resize: 'both',
        });

        // Ensure the content container allows for scrolling overflow content
        Object.assign(this.contentContainer.style, {
            overflowY: 'auto',
            maxHeight: '565px', // Adjust based on titleBar height to prevent overflow
        });
        this.adjustContentContainerSize();
    }
    adjustContentContainerSize() {
        // Calculate available height for content container
        const titleBarHeight = this.titleBar.offsetHeight;
        const consoleHeight = this.element.offsetHeight;
        const availableHeight = consoleHeight - titleBarHeight;

        // Update content container's maxHeight to fill available space
        this.contentContainer.style.maxHeight = `${availableHeight}px`;
    }
    createTitleBar() {
        const bar = document.createElement('div');
        bar.textContent = 'Console';
        Object.assign(bar.style, {
            padding: '5px',
            cursor: 'move',
            backgroundColor: '#333',
            borderTopLeftRadius: '5px',
            borderTopRightRadius: '5px',
            userSelect: 'none',
        });
        return bar;
    }

    createContentContainer() {
        const container = document.createElement('div');
        return container;
    }

    addEventListeners() {
        this.titleBar.addEventListener('mousedown', (e) => {
            // Mark as dragging
            this.dragging = true;

            // Record the initial mouse position
            this.prevX = e.clientX;
            this.prevY = e.clientY;

            if (!this.element.style.left || !this.element.style.top) {
                // Calculate initial left and top based on current position
                const rect = this.element.getBoundingClientRect();
                this.element.style.right = ''; // Clear 'right' since we're switching to 'left/top' positioning
                this.element.style.bottom = ''; // Clear 'bottom' as well

                // Set initial left and top based on the element's current position
                this.element.style.left = `${rect.left}px`;
                this.element.style.top = `${rect.top}px`;
            }
        });

        document.addEventListener('mousemove', (e) => {
            if (!this.dragging) return;
            const dx = e.clientX - this.prevX;
            const dy = e.clientY - this.prevY;

            const { style } = this.element;
            style.left = `${parseInt(style.left || 0, 10) + dx}px`;
            style.top = `${parseInt(style.top || 0, 10) + dy}px`;

            this.prevX = e.clientX;
            this.prevY = e.clientY;
        });

        document.addEventListener('mouseup', () => {
            this.dragging = false;
            this.adjustContentContainerSize();
        });
    }

    addMenuButton() {
        const menu = document.querySelector(".comfy-menu");
        // Create and append the toggle button for the floating console
        const consoleToggleButton = document.createElement("button");
        consoleToggleButton.textContent = "Toggle Console";
        consoleToggleButton.onclick = () => {
            // Check if the console is currently visible and toggle accordingly
            if (floatingConsole.isVisible()) {
                floatingConsole.hide();
                consoleToggleButton.textContent = "Show Console"; // Update button text
            } else {
                floatingConsole.show();
                consoleToggleButton.textContent = "Hide Console"; // Update button text
            }
        }
        menu.append(consoleToggleButton);
    }

    show() {
        this.element.style.display = 'block';
    }

    hide() {
        this.element.style.display = 'none';
    }

    isVisible() {
        return this.element.style.display !== 'none';
    }

    log(message) {
        const msgElement = document.createElement('div');
        msgElement.textContent = message;
        this.contentContainer.appendChild(msgElement);
        this.contentContainer.scrollTop = this.contentContainer.scrollHeight; // Auto-scroll to bottom
    }

    clear() {
        this.contentContainer.innerHTML = '';
    }
}

// Instantiate the floating console
const floatingConsole = new FloatingConsole();

// Extend the app plugin to handle console_output events
app.registerExtension({
    name: "consoleOutput",
    init() {
        api.addEventListener('console_output', (event) => {
            floatingConsole.log(event.detail.message);
        });
    }
});


