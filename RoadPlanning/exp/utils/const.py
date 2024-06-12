NUM_OF_BUILD_WAYS = 50
BUILD_WAYS_UPPER_BOUND = 30

NUM_OF_BUILD_WAYS = 50

conf = {}
conf["num_hiddens"] = 64
conf["num_layers"] = 2
conf["output_noise"] = False
conf["rand_prior"] = True
conf["verbose"] = False
conf["l1"] = 3e-3
conf["lr"] = 3e-2
conf["num_epochs"] = 100
BO_CONF = conf

# routing listening host
HOST = "localhost:52901"

# SA
INIT_TEMP = 150
ALPHA = 0.98
LOCAL_SEARCH_TIMES = 3
FINAL_TEMP = 100
