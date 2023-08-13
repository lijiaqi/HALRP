import os.path as osp
import importlib


class TinyImagenetArgs:
    def __init__(
        self,
        data_path,
        loader_type,
        pc_valid,
        increment,
        class_order,
        batch_size_train,
        batch_size_test,
        workers,
        pin_memory,
        dataset="tinyimagenet",
    ):
        self.data_path = data_path
        self.loader_type = loader_type
        self.pc_valid = pc_valid
        self.increment = increment
        self.class_order = class_order
        self.batch_size_train = batch_size_train
        self.batch_size_test = batch_size_test
        self.workers = workers
        self.pin_memory = pin_memory
        self.dataset = dataset


def get_loader(
    data_dir,
    seed=1,
    fixed_order=False,  # for task_order, not class_order
    pc_valid=0.1,
    batch_size=10,
    num_workers=4,
    pin_memory=False,
    loader_type="class_incremental_loader",
    increment=5,
):
    # create tiny imagenet arguments
    data_path = osp.join(data_dir, "tiny-imagenet-200/")
    class_order = "old"
    if not fixed_order:
        print("Using random task order determined by seed={}".format(seed))
    else:
        raise "Not implemented for fixed_order={}".format(fixed_order)
    options = TinyImagenetArgs(
        data_path,
        loader_type=loader_type,
        pc_valid=pc_valid,
        increment=increment,
        class_order=class_order,
        batch_size_train=batch_size,
        batch_size_test=batch_size,
        workers=num_workers,
        pin_memory=pin_memory,
        dataset="tinyimagenet",
    )
    # loader
    Loader = importlib.import_module("loader_tiny." + options.loader_type)
    loader = Loader.IncrementalLoader(opt=options, seed=seed)
    n_inputs, n_outputs, n_tasks, input_size = loader.get_dataset_info()
    print(
        "Dataset inform: num_outputs={}, n_tasks={}, input_size={}".format(
            n_outputs, n_tasks, input_size
        )
    )

    train_loader_splits = {}
    valid_loader_splits = loader.get_tasks("test") # without data augmentation
    test_loader_splits = loader.get_tasks("val") # without data augmentation
    task_output_space = {}
    tasks_info = {}
    for t in range(n_tasks):
        task_info, train_loader, _, _ = loader.new_task()
        task_info["n_train_data"] = len(train_loader.dataset)
        task_info["n_test_data"] = len(test_loader_splits[t].dataset)
        task_info["n_val_data"] = len(valid_loader_splits[t].dataset)
        print("Task-{}: {}".format(t, task_info))


        train_loader_splits[t] = train_loader
        task_output_space[t] = task_info["increment"]
        tasks_info[t] = task_info

    return (
        loader,
        train_loader_splits,
        valid_loader_splits,
        test_loader_splits,
        task_output_space,
        tasks_info,
    )


if __name__ == "__main__":
    (
        loader,
        train_loader_splits,
        valid_loader_splits,
        test_loader_splits,
        task_output_space,
        tasks_info,
    ) = get_loader(data_dir="../data")
    print(tasks_info)
    import numpy as np
    print(np.unique(train_loader_splits[2].dataset.y))
    print(np.unique(valid_loader_splits[2].dataset.y))
    print(np.unique(test_loader_splits[2].dataset.y))
