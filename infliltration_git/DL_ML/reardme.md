# get_items.py
函数：get_model(model_name, n_channels, n_classes, data_mode)
输入：
    model_name-模型的名字；
    n_channels-通道；
    n_classes-类别；
    data_mode-数据类型的标志
输出：
    model
函数：get_loss(loss_name, loss_dict = None)
输入：
    loss_name-损失函数的类型；
    loss_dict-损失函数的参数；
输出：
    loss
函数：add_weight_decay(model, weight_decay=1e-5, skip_list=())
输入：
    model-模型；
    weight_decay-衰减系数；
    skip_list-需要跳过的参数
输出：
    需要衰减系数的参数
函数：get_optimizer(optimizer_name, optimizer_opt, lr, model)
输入：
    optimizer_name-优化器的名称；
    optimizer_opt-优化器的参数；
    lr-学习率
    model-模型
输出：
    优化器
函数：get_lr_scheduler(lr_scheduler_name, optimizer, lr_scheduler_opt = {
    #'milestones': [60-64, 90-64],
    'milestones': [40, 80, 120],
    'gamma': 0.1,})
输入：
    lr_scheduler_name-学习率的选择；
    optimizer-优化器的参数；
    lr-学习率
    model-模型
输出：
    优化器