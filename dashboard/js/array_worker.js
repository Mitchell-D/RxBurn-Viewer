
self.onmessage = async (a) => {
    const {type, id, args} = a.data;
    console.log(type, id, args);
    try {
        if (type === "load-array") {
            const {key, buffer, width, height, ntimes} = args;
            const x = new Uint16Array(buffer);
            console.log("loading array");
            console.log(key);
        } else if (type === "delete-array") {
            console.log("")
        } else if (type === "get-mask") {
        } else if (type === "get-rgb") {
        } else {
            console.error("unrecognized type:", type);
        }
    } catch (error) {
        self.postMessage({id, success:false, error:error.toString()});
    }
}
