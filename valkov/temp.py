def train_model(model, train_data, train_labels, test_data=None, test_labels=None):
    """_summary_

    Args:
        model (_type_): _description_
        train_data (_type_): _description_
        train_labels (_type_): _description_
        test_data (_type_, optional): _description_. Defaults to None.
        test_labels (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    return model.eval(), train_hist, test_hist