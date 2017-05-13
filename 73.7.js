//<![CDATA[

// a few things don't have var in front of them - they update already existing variables the game needs
lanesSide = 4;
patchesAhead = 20;
patchesBehind = 4;
trainIterations = 60000;

var num_inputs = (lanesSide * 2 + 1) * (patchesAhead + patchesBehind);
var num_actions = 5;
var temporal_window = 0;
var network_size = num_inputs * temporal_window + num_actions * temporal_window + num_inputs;

var layer_defs = [];
layer_defs.push({
    type: 'input',
    out_sx: 1,
    out_sy: 1,
    out_depth: network_size
});
layer_defs.push({
    type: 'fc',
    num_neurons: 100,
    activation: 'relu'
});
layer_defs.push({
    type: 'fc',
    num_neurons: 100,
    activation: 'relu'
});
layer_defs.push({
    type: 'regression',
    num_neurons: num_actions
});

var tdtrainer_options = {
    method: 'adagrad',
    batch_size: 100,
    l2_decay: 0.002
};

var opt = {};
opt.temporal_window = temporal_window;
opt.experience_size = 10000;
opt.start_learn_threshold = 100;
opt.gamma = 0.9;
opt.learning_steps_total = 10000;
opt.learning_steps_burnin = 1000;
opt.epsilon_min = 0.02;
opt.epsilon_test_time = 0.04;
opt.layer_defs = layer_defs;
opt.tdtrainer_options = tdtrainer_options;

brain = new deepqlearn.Brain(num_inputs, num_actions, opt);

learn = function (state, lastReward) {
    brain.backward(lastReward);
    var action = brain.forward(state);

    draw_net();
    draw_stats();

    return action;
}

//]]>