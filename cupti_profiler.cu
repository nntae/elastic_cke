#include "cupti_profiler.h"

// Static data

bool PRINT_VALUES = true;
bool MAX_VERBOSITY = false;

CUcontext m_context = 0;
CUdevice m_device = 0;
CUpti_SubscriberHandle m_subscriber;
MetricData_t m_metricData;
CUpti_MetricID m_metricId;

CUpti_EventID *m_eventId;
uint64_t *m_numEvents;

uint64_t m_kernelDuration;

FILE *m_fp = stderr;
char m_metricName[255];

// Routines

void CUPTIAPI
getMetricValueCallback(void *userdata, CUpti_CallbackDomain domain,
                       CUpti_CallbackId cbid, const CUpti_CallbackData *cbInfo)
{
	MetricData_t *metricData = (MetricData_t*)userdata;
	unsigned int i, j, k;

	// This callback is enabled only for launch so we shouldn't see anything else.
	if ((cbid != CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020) &&
			(cbid != CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000))
	{
		printf("%s:%d: unexpected cbid %d\n", __FILE__, __LINE__, cbid);
		exit(-1);
	}

	// on entry, enable all the event groups being collected this pass,
	// for metrics we collect for all instances of the event

	if (cbInfo->callbackSite == CUPTI_API_ENTER) 
	{
		cudaDeviceSynchronize();
		CUPTI_CALL(cuptiSetEventCollectionMode(cbInfo->context,	CUPTI_EVENT_COLLECTION_MODE_KERNEL));

		for (i = 0; i < metricData->eventGroups->numEventGroups; i++) 
		{
			uint32_t all = 1;
			CUPTI_CALL(cuptiEventGroupSetAttribute(metricData->eventGroups->eventGroups[i],
                                             CUPTI_EVENT_GROUP_ATTR_PROFILE_ALL_DOMAIN_INSTANCES,
                                             sizeof(all), &all));
			CUPTI_CALL(cuptiEventGroupEnable(metricData->eventGroups->eventGroups[i]));
		}
	}

	// on exit, read and record event values
	if (cbInfo->callbackSite == CUPTI_API_EXIT) 
	{
		cudaDeviceSynchronize();
		// for each group, read the event values from the group and record in metricData
		for (i = 0; i < metricData->eventGroups->numEventGroups; i++) 
		{
			CUpti_EventGroup group = metricData->eventGroups->eventGroups[i];
			CUpti_EventDomainID groupDomain;
			uint32_t numEvents, numInstances, numTotalInstances;
			CUpti_EventID *eventIds;
			size_t groupDomainSize = sizeof(groupDomain);
			size_t numEventsSize = sizeof(numEvents);
			size_t numInstancesSize = sizeof(numInstances);
			size_t numTotalInstancesSize = sizeof(numTotalInstances);
			uint64_t *values, normalized, sum;
			size_t valuesSize, eventIdsSize;

			CUPTI_CALL(cuptiEventGroupGetAttribute(group,
                                             CUPTI_EVENT_GROUP_ATTR_EVENT_DOMAIN_ID,
                                             &groupDomainSize, &groupDomain));
			CUPTI_CALL(cuptiDeviceGetEventDomainAttribute(metricData->device, groupDomain,
                                                    CUPTI_EVENT_DOMAIN_ATTR_TOTAL_INSTANCE_COUNT,
                                                    &numTotalInstancesSize, &numTotalInstances));
			CUPTI_CALL(cuptiEventGroupGetAttribute(group,
                                             CUPTI_EVENT_GROUP_ATTR_INSTANCE_COUNT,
                                             &numInstancesSize, &numInstances));
			CUPTI_CALL(cuptiEventGroupGetAttribute(group,
                                             CUPTI_EVENT_GROUP_ATTR_NUM_EVENTS,
                                             &numEventsSize, &numEvents));
			eventIdsSize = numEvents * sizeof(CUpti_EventID);
			eventIds = (CUpti_EventID *)malloc(eventIdsSize);
			CUPTI_CALL(cuptiEventGroupGetAttribute(group,
                                             CUPTI_EVENT_GROUP_ATTR_EVENTS,
                                             &eventIdsSize, eventIds));
			valuesSize = sizeof(uint64_t) * numInstances;
			values = (uint64_t *)malloc(valuesSize);
			
			for (j = 0; j < numEvents; j++) 
			{
				CUPTI_CALL(cuptiEventGroupReadEvent(group, CUPTI_EVENT_READ_FLAG_NONE,
                                            eventIds[j], &valuesSize, values));
				if (metricData->eventIdx >= metricData->numEvents) 
				{
					fprintf(stderr, "error: too many events collected, metric expects only %d\n", (int)metricData->numEvents);
					exit(-1);
				}

				// sum collect event values from all instances
				sum = 0;
				for (k = 0; k < numInstances; k++)
					sum += values[k];

				// normalize the event value to represent the total number of
				// domain instances on the device
				normalized = (sum * numTotalInstances) / numInstances;

				metricData->eventIdArray[metricData->eventIdx] = eventIds[j];
				metricData->eventValueArray[metricData->eventIdx] = normalized;
				metricData->eventIdx++;

				// print collected value
				if ( PRINT_VALUES )
				{
					char eventName[128];
					size_t eventNameSize = sizeof(eventName) - 1;
					CUPTI_CALL(cuptiEventGetAttribute(eventIds[j], CUPTI_EVENT_ATTR_NAME,
												&eventNameSize, eventName));
					
					eventName[127] = '\0';
					fprintf(m_fp, "%s, %s, %llu, %llu", m_metricName, eventName, (unsigned long long)eventIds[j], (unsigned long long)sum);
					fprintf(m_fp, "%u, %u, %llu", numTotalInstances, numInstances, (unsigned long long)normalized);
					for (k = 0; k < numInstances; k++) 
						fprintf(m_fp, "%llu, ", (unsigned long long)values[k]);
					fprintf(m_fp, "\n");

				}
			}
			free(values);
		}
		for (i = 0; i < metricData->eventGroups->numEventGroups; i++)
			CUPTI_CALL(cuptiEventGroupDisable(metricData->eventGroups->eventGroups[i]));
	}
}

static void CUPTIAPI
bufferRequested(uint8_t **buffer, size_t *size, size_t *maxNumRecords)
{
  uint8_t *rawBuffer;

  *size = 16 * 1024;
  rawBuffer = (uint8_t *)malloc(*size + ALIGN_SIZE);

  *buffer = ALIGN_BUFFER(rawBuffer, ALIGN_SIZE);
  *maxNumRecords = 0;

  if (*buffer == NULL) {
    printf("Error: out of memory\n");
    exit(-1);
  }
}

static void CUPTIAPI
bufferCompleted(CUcontext ctx, uint32_t streamId, uint8_t *buffer, size_t size, size_t validSize)
{
  CUpti_Activity *record = NULL;
  CUpti_ActivityKernel4 *kernel;

  //since we launched only 1 kernel, we should have only 1 kernel record
  CUPTI_CALL(cuptiActivityGetNextRecord(buffer, validSize, &record));

  kernel = (CUpti_ActivityKernel4 *)record;
  if (kernel->kind != CUPTI_ACTIVITY_KIND_KERNEL) {
    fprintf(stderr, "Error: expected kernel activity record, got %d\n", (int)kernel->kind);
    exit(-1);
  }

  m_kernelDuration = kernel->end - kernel->start;
  free(buffer);
}

std::vector<std::string> 
init_cupti_profiler( const int device_num )
{
	int deviceCount;
	char deviceName[32];

	// Make sure activity is enabled before any CUDA API
	CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL));

	// Init CUDA and create context
	DRIVER_API_CALL(cuInit(0));
	DRIVER_API_CALL(cuDeviceGetCount(&deviceCount));
	if ( deviceCount == 0 )
	{
		printf("There is no device supporting CUDA.\n");
		exit(-1);
	}
	if ( device_num >= deviceCount )
	{
		printf("Device %d does not exist. Device count is %d\n", device_num, deviceCount);
		exit(-2);
	}
	DRIVER_API_CALL(cuDeviceGet(&m_device, device_num));
	DRIVER_API_CALL(cuDeviceGetName(deviceName, 32, m_device));
	printf("CUDA Device Name: %s\n", deviceName);
	DRIVER_API_CALL(cuCtxCreate(&m_context, 0, m_device));
	m_kernelDuration = 0;

	return available_metrics_cupti_profiler( m_device,true);
	
}

void
start_kernelduration_cupti_profiler()
{
	CUPTI_CALL(cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted));
}

uint64_t 
end_kernelduration_cupti_profiler()
{
	cudaDeviceSynchronize();
	CUPTI_CALL(cuptiActivityFlushAll(0));
	return m_kernelDuration;
}

CUpti_EventGroupSets *
start_cupti_profiler(	const char *metricName )
{
	CUpti_EventGroupSets *passData;

	sprintf(m_metricName, "%s", metricName);
	
	// setup launch callback for event collection
	CUPTI_CALL(cuptiSubscribe(&m_subscriber, (CUpti_CallbackFunc)getMetricValueCallback, &m_metricData));
	CUPTI_CALL(cuptiEnableCallback(1, m_subscriber, CUPTI_CB_DOMAIN_RUNTIME_API,
										CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020));
	CUPTI_CALL(cuptiEnableCallback(1, m_subscriber, CUPTI_CB_DOMAIN_RUNTIME_API,
										CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000));

	// allocate space to hold all the events needed for the metric
	CUPTI_CALL(cuptiMetricGetIdFromName(m_device, metricName, &m_metricId));
	CUPTI_CALL(cuptiMetricGetNumEvents(m_metricId, &m_metricData.numEvents));

	m_metricData.device = m_device;
	m_eventId = (CUpti_EventID *)malloc(m_metricData.numEvents * sizeof(CUpti_EventID));
	m_metricData.eventIdArray = m_eventId;
	m_numEvents = (uint64_t *)malloc(m_metricData.numEvents * sizeof(uint64_t));
	m_metricData.eventValueArray = m_numEvents;
	m_metricData.eventIdx = 0;

	// get the number of passes required to collect all the events
	// needed for the metric and the event groups for each pass
	CUPTI_CALL(cuptiMetricCreateEventGroupSets(m_context, sizeof(m_metricId), &m_metricId, &passData));

	return passData;
}

void
advance_cupti_profiler( CUpti_EventGroupSets *passData, int pass )
{
	m_metricData.eventGroups = passData->sets + pass;
}

void
stop_cupti_profiler( bool getvalue )
{
	CUpti_MetricValue metricValue;
//	printf("Kernel duration %llu\n", (unsigned long long) m_kernelDuration);
	
	// use all the collected events to calculate the metric value
	if ( getvalue )
	{
		CUPTI_CALL(cuptiMetricGetValue(m_device, m_metricId,
									m_metricData.numEvents * sizeof(CUpti_EventID),
									m_metricData.eventIdArray,
									m_metricData.numEvents * sizeof(uint64_t),
									m_metricData.eventValueArray,
									m_kernelDuration, &metricValue));

		// print metric value, we format based on the value kind
		{
			CUpti_MetricValueKind valueKind;
			size_t valueKindSize = sizeof(valueKind);
			CUPTI_CALL(cuptiMetricGetAttribute(m_metricId, CUPTI_METRIC_ATTR_VALUE_KIND,
													&valueKindSize, &valueKind));
			switch (valueKind) 
			{
				case CUPTI_METRIC_VALUE_KIND_DOUBLE:
					// printf("Metric %f\n", metricValue.metricValueDouble);
					fprintf(m_fp, "%s, %f\n", m_metricName, metricValue.metricValueDouble);
					break;
				case CUPTI_METRIC_VALUE_KIND_UINT64:
					// printf("Metric %llu\n", (unsigned long long)metricValue.metricValueUint64);
					fprintf(m_fp, "%s, %llu\n", m_metricName, (unsigned long long)metricValue.metricValueUint64);
					break;
				case CUPTI_METRIC_VALUE_KIND_INT64:
					// printf("Metric %lld\n", (long long)metricValue.metricValueInt64);
					fprintf(m_fp, "%s, %lld\n", m_metricName, (long long)metricValue.metricValueInt64);
					break;
				case CUPTI_METRIC_VALUE_KIND_PERCENT:
					// printf("Metric %f%%\n", metricValue.metricValuePercent);
					fprintf(m_fp, "%s, %f\n", m_metricName, metricValue.metricValuePercent);
					break;
				case CUPTI_METRIC_VALUE_KIND_THROUGHPUT:
					fprintf(m_fp, "%s, %llu\n", m_metricName, (unsigned long long)metricValue.metricValueThroughput);
					break;
				case CUPTI_METRIC_VALUE_KIND_UTILIZATION_LEVEL:
					// printf("Metric utilization level %u\n", (unsigned int)metricValue.metricValueUtilizationLevel);
					fprintf(m_fp, "%s, %u\n", m_metricName, (unsigned int)metricValue.metricValueUtilizationLevel);
					break;
				default:
					fprintf(stderr, "error: unknown value kind\n");
					exit(-1);
			}
		}
	}

	// Unsubscribe and free data
	CUPTI_CALL(cuptiUnsubscribe(m_subscriber));
	//CUPTI_CALL(cuptiEventGroupSetsDestroy(m_metric_pass_data));
	free(m_eventId);
	free(m_numEvents);
}

std::vector<std::string> 
available_metrics_cupti_profiler(	CUdevice device,
									bool print_names)
{
	std::vector<std::string> metric_names;
	uint32_t numMetric;
	size_t size;
	char metricName[__CUPTI_PROFILER_NAME_SHORT];
	CUpti_MetricValueKind metricKind;
	CUpti_MetricID *metricIdArray;

	CUPTI_CALL(cuptiDeviceGetNumMetrics(device, &numMetric));

	size = sizeof(CUpti_MetricID) * numMetric;
	metricIdArray = (CUpti_MetricID*) malloc(size);
	if(NULL == metricIdArray) 
	{
		printf("Memory could not be allocated for metric array");
		exit(-1);
	}

	CUPTI_CALL(cuptiDeviceEnumMetrics(device, &size, metricIdArray));

	if ( print_names )
		printf("%d available metrics:\n", numMetric);

	for (int i = 0; i < numMetric; i++) 
	{
		size = __CUPTI_PROFILER_NAME_SHORT;
		CUPTI_CALL(cuptiMetricGetAttribute(metricIdArray[i], CUPTI_METRIC_ATTR_NAME,
												&size, (void *)& metricName));
		size = sizeof(CUpti_MetricValueKind);
		CUPTI_CALL(cuptiMetricGetAttribute(metricIdArray[i], CUPTI_METRIC_ATTR_VALUE_KIND,
												&size, (void *)& metricKind));

		if ( (metricKind == CUPTI_METRIC_VALUE_KIND_THROUGHPUT) 
			|| (metricKind == CUPTI_METRIC_VALUE_KIND_UTILIZATION_LEVEL) )
		{
			if ( print_names && MAX_VERBOSITY )
				printf("Metric %s cannot be profiled as metric requires GPU time duration for kernel run.\n", metricName);
		}
		else 
		{
			metric_names.push_back(metricName);
			if ( print_names )
				if ( i > 0 )
					printf(", %s", metricName);
				else
					printf("%s", metricName);
		}												

	}

	if ( print_names )
		printf("\n %lu metrics will be profiled\n", metric_names.size());

	free(metricIdArray);
	return std::move(metric_names);
}

FILE *
open_metric_file( const char *name )
{
	m_fp = fopen(name, "a");
	return m_fp;
}

void
close_metric_file()
{
	fclose(m_fp);
}
