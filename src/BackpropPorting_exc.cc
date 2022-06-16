/**
 * @author Niccolò Nicolosi <niccolo.nicolosi@mail.polimi.it>
 */

#include "BackpropPorting_exc.h"

BackpropPorting_exc::BackpropPorting_exc(std::string const & name,
		std::string const & recipe,
		RTLIB_Services_t *rtlib,
        int argc,
        char *argv[]) :
	BbqueEXC(name, recipe, rtlib, RTLIB_LANG_CPP) 
{
	cout << "New BackpropPorting_exc::BackpropPorting_exc() UID=" << GetUniqueID() << endl;

    if(argc!=2) {
        fprintf(stderr, "usage: backprop <num of input elements>\n");
        exit(0);
    }

    layer_size = atoi(argv[1]);
}

RTLIB_ExitCode_t BackpropPorting_exc::onSetup()
{
	cout << "BackpropPorting_exc::onSetup()" << endl;

    int seed;

    seed = 7;
    bpnn_initialize(seed);

    net = bpnn_create(layer_size, 16, 1); // (16, 1 can not be changed)
    printf("Input layer size : %d\n", layer_size);

    work_step = 0;
    counter = 1;

    in = net->input_n;
    hid = net->hidden_n;
    out = net->output_n;

    pid = getpid();

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///CODICE NECESSARIO PER L'INTEGRAZIONE CON DROM////////////////////////////////////////////////////////////////////
    /*cpu_set_t cpu_set;
    CPU_ZERO(&cpu_set);
    if(DLB_Init(0, &cpu_set, "--drom") == DLB_SUCCESS) printf("DLB correctly initialized\n");
    else printf("DLB initialization FAILED\n");*/
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    printf("Starting training\n");

	return RTLIB_OK;
}

RTLIB_ExitCode_t BackpropPorting_exc::onConfigure(int8_t awm_id)
{
	GetAssignedResources(PROC_ELEMENT, proc_quota);
	GetAssignedResources(PROC_NR, proc_nr);
	GetAssignedResources(GPU, acc);
	GetAssignedResources(ACCELERATOR, acc);

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///CODICE NECESSARIO PER L'INTEGRAZIONE CON DROM////////////////////////////////////////////////////////////////////
    /*cpu_set_t cpu_set;
    CPU_ZERO(&cpu_set);
    cout << "Updating available processors: " << DLB_Strerror(DLB_PollDROM(&proc_nr, &cpu_set)) << endl;*/
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    cout << "BackpropPorting_exc::onConfigure(): proc_nr = " << proc_nr << ", proc_quota = " << proc_quota << endl;

    cout << endl << endl;

	return RTLIB_OK;
}

RTLIB_ExitCode_t BackpropPorting_exc::onRun() {
	RTLIB_WorkingModeParams_t const wmp = WorkingModeParams();

    if (work_step == 6) return RTLIB_EXC_WORKLOAD_NONE;

	cout << "BackpropPorting_exc::onRun(): Hello AEM! cycle="<< Cycles() << endl;

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    printf("Performing CPU computation: step %d\n", work_step);

    switch(work_step) {

        case 0:
            if(load(net, layer_size, &counter) == 1) {

                work_step += 1;
                /*printf("input_units:\n");
                for(int n = 1; n <= in; n++) printf("%d: %f\n", n, net->input_units[n]);*/
            }
            break;
        case 1:
            if(bpnn_layerforward(net->input_units, net->hidden_units, net->input_weights, in, hid, proc_nr, &counter) == 1) {

                work_step += 1;
                /*printf("hidden_units:\n");
                for(int n = 1; n <= hid; n++) printf("%d: %f\n", n, net->hidden_units[n]);*/
            }
            break;
        case 2:
            if(bpnn_layerforward(net->hidden_units, net->output_units, net->hidden_weights, hid, out, proc_nr, &counter) == 1) {

                work_step += 1;
                /*printf("output_units:\n");
                for(int n = 1; n <= out; n++) printf("%d: %f\n", n, net->output_units[n]);*/
            }
            break;
        case 3:
            bpnn_output_error(net->output_delta, net->target, net->output_units, out, &out_err);
            bpnn_hidden_error(net->hidden_delta, hid, net->output_delta, out, net->hidden_weights, net->hidden_units, &hid_err);
            work_step += 1;
            /*printf("output_delta:\n");
            for(int n = 1; n <= out; n++) printf("%d: %f\n", n, net->output_delta[n]);
            printf("out_err: %f\n", out_err);
            printf("hidden_delta:\n");
            for(int n = 1; n <= hid; n++) printf("%d: %f\n", n, net->hidden_delta[n]);
            printf("hid_err: %f\n", hid_err);*/
            break;
        case 4:
            if(bpnn_adjust_weights(net->output_delta, out, net->hidden_units, hid, net->hidden_weights, net->hidden_prev_weights, proc_nr, &counter) == 1) {

                work_step += 1;
                /*printf("hidden_weights:\n");
                for(int j = 1; j <= out; j++) {

                    for(int k = 0; k <= hid; k++) printf("[%d, %d]: %f\n", k, j, net->hidden_weights[k][j]);
                }
                printf("hidden_prev_weights:\n");
                for(int j = 1; j <= out; j++) {

                    for(int k = 0; k <= hid; k++) printf("[%d, %d]: %f\n", k, j, net->hidden_prev_weights[k][j]);
                }*/
            }
            break;
        case 5:
            if(bpnn_adjust_weights(net->hidden_delta, hid, net->input_units, in, net->input_weights, net->input_prev_weights, proc_nr, &counter) == 1) {

                work_step += 1;
                /*printf("input_weights:\n");
                for(int j = 1; j <= hid; j++) {

                    for(int k = 0; k <= in; k++) printf("[%d, %d]: %f\n", k, j, net->input_weights[k][j]);
                }
                printf("input_prev_weights:\n");
                for(int j = 1; j <= hid; j++) {

                    for(int k = 0; k <= in; k++) printf("[%d, %d]: %f\n", k, j, net->input_prev_weights[k][j]);
                }*/
            }
            break;
    }

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Cycle time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms" << std::endl;

	return RTLIB_OK;
}

RTLIB_ExitCode_t BackpropPorting_exc::onMonitor()
{
	cout << "BackpropPorting_exc::onMonitor(): CPS=" << GetCPS() << endl << endl;

	return RTLIB_OK;
}

RTLIB_ExitCode_t BackpropPorting_exc::onSuspend()
{
	cout << "BackpropPorting_exc::onMonitor()" << GetCPS() << endl;

	return RTLIB_OK;
}

RTLIB_ExitCode_t BackpropPorting_exc::onRelease()
{
	cout << "BackpropPorting_exc::onRelease()" << endl;

    bpnn_free(net);
    printf("Training done\n");

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///CODICE NECESSARIO PER L'INTEGRAZIONE CON DROM////////////////////////////////////////////////////////////////////
    /*if(DLB_Finalize() == DLB_SUCCESS) printf("DLB correctly finalized\n");
    else printf("DLB finalization FAILED\n");*/
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    return RTLIB_OK;
}
