import execution, nodes
orig_exec = execution.map_node_over_list

def map_node_over_list(obj, input_data_all, func, allow_interrupt=False):
    try:
        # check if node wants the lists
        input_is_list = False
        if hasattr(obj, "INPUT_IS_LIST"):
            input_is_list = obj.INPUT_IS_LIST

        if len(input_data_all) == 0:
            max_len_input = 0
        else:
            max_len_input = max([len(x) for x in input_data_all.values() if x is not None])
        # get a slice of inputs, repeat last input when list isn't long enough
        def slice_dict(d, i):
            d_new = dict()
            for k, v in d.items():
                if v is None:  # Skip if any of the values is None
                    return None
                d_new[k] = v[i if len(v) > i else -1]
            return d_new

        results = []

        for k, v in input_data_all.items():
            if v == "skip":
                print("[deforum] Skipping execution of", obj)

                return []
        if input_is_list:
            if allow_interrupt:
                nodes.before_node_execution()
            results.append(getattr(obj, func)(**input_data_all))
        elif max_len_input == 0:
            if allow_interrupt:
                nodes.before_node_execution()
            results.append(getattr(obj, func)())
        else:
            for i in range(max_len_input):
                sliced_input = slice_dict(input_data_all, i)
                if sliced_input is None:  # Skip this iteration if there's a None value after slicing
                    continue
                if allow_interrupt:
                    nodes.before_node_execution()
                for k, v in sliced_input.items():
                    if v == "skip":
                        print("[deforum] Skipping execution of", obj)
                        return []
                results.append(getattr(obj, func)(**sliced_input))
        return results
    except:
        print("[deforum] Executor HiJack Failed and was deactivated, please report the issue on GitHub")
        execution.map_node_over_list = orig_exec
        return orig_exec(obj, input_data_all, func, allow_interrupt)

execution.map_node_over_list = map_node_over_list

print("[deforum] Execution HiJack Active")