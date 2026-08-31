#include "lm_air_ble.h"

#include <assert.h>
#include <string.h>
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "host/ble_att.h"
#include "host/ble_gap.h"
#include "host/ble_gatt.h"
#include "host/ble_hs.h"
#include "host/ble_uuid.h"
#include "nimble/nimble_port.h"
#include "nimble/nimble_port_freertos.h"
#include "services/gap/ble_svc_gap.h"
#include "services/gatt/ble_svc_gatt.h"

#include "lm_air_ble_frag.h"
#include "lm_air_metrics.h"
#include "lm_air_radio.h"

static const char *TAG = "lm_ble";
static uint8_t s_addr_type;
static uint16_t s_conn_handle = BLE_HS_CONN_HANDLE_NONE;
static uint16_t s_value_handle;
static bool s_subscribed;
static uint16_t s_message_id;
static lm_air_ble_reassembly_t s_reassembly;

static const ble_uuid128_t s_service_uuid =
    BLE_UUID128_INIT(0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe,
                     0x01, 0x00, 0x4c, 0x4d, 0x41, 0x49, 0x52, 0x01);
static const ble_uuid128_t s_char_uuid =
    BLE_UUID128_INIT(0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe,
                     0x02, 0x00, 0x4c, 0x4d, 0x41, 0x49, 0x52, 0x01);

static int gatt_access(uint16_t conn_handle,
                       uint16_t attr_handle,
                       struct ble_gatt_access_ctxt *ctxt,
                       void *arg)
{
    (void)conn_handle;
    (void)attr_handle;
    (void)arg;
    if (ctxt->op != BLE_GATT_ACCESS_OP_WRITE_CHR) {
        return BLE_ATT_ERR_UNLIKELY;
    }
    uint8_t fragment[CONFIG_LM_MAX_FRAME_SIZE + LM_AIR_BLE_FRAG_HEADER_SIZE];
    uint16_t len = 0;
    const int rc = ble_hs_mbuf_to_flat(ctxt->om, fragment, sizeof(fragment), &len);
    if (rc != 0) {
        lm_air_metric_add(LM_METRIC_BAD_FRAME, 1);
        return BLE_ATT_ERR_INVALID_ATTR_VALUE_LEN;
    }
    lm_air_packet_t packet;
    const lm_air_ble_result_t result =
        lm_air_ble_reassembly_ingest(&s_reassembly, fragment, len, &packet);
    if (result == LM_AIR_BLE_COMPLETE) {
        packet.flags = LM_AIR_PAYLOAD_PUBLIC_CODEC;
        if (lm_air_radio_ingest(LM_AIR_LINK_BLE, packet.data, packet.len,
                                packet.flags, 0) == ESP_OK) {
            lm_air_metric_add(LM_METRIC_BLE_RX, 1);
        }
    } else if (result == LM_AIR_BLE_REJECTED) {
        lm_air_metric_add(LM_METRIC_BAD_FRAME, 1);
        return BLE_ATT_ERR_UNLIKELY;
    }
    return 0;
}

static const struct ble_gatt_svc_def s_services[] = {
    {
        .type = BLE_GATT_SVC_TYPE_PRIMARY,
        .uuid = &s_service_uuid.u,
        .characteristics = (struct ble_gatt_chr_def[]) {{
            .uuid = &s_char_uuid.u,
            .access_cb = gatt_access,
            .val_handle = &s_value_handle,
            .flags = BLE_GATT_CHR_F_WRITE | BLE_GATT_CHR_F_WRITE_NO_RSP |
                     BLE_GATT_CHR_F_NOTIFY,
        }, {0}},
    },
    {0},
};

static void advertise(void);

static int gap_event(struct ble_gap_event *event, void *arg)
{
    (void)arg;
    switch (event->type) {
    case BLE_GAP_EVENT_CONNECT:
        if (event->connect.status == 0) {
            s_conn_handle = event->connect.conn_handle;
        } else {
            advertise();
        }
        return 0;
    case BLE_GAP_EVENT_DISCONNECT:
        s_conn_handle = BLE_HS_CONN_HANDLE_NONE;
        s_subscribed = false;
        lm_air_ble_reassembly_reset(&s_reassembly);
        advertise();
        return 0;
    case BLE_GAP_EVENT_SUBSCRIBE:
        if (event->subscribe.attr_handle == s_value_handle) {
            s_subscribed = event->subscribe.cur_notify != 0;
        }
        return 0;
    default:
        return 0;
    }
}

/* A legacy advertisement PDU carries at most 31 bytes of AD structures. Flags
 * cost 3 and a complete 128-bit service UUID costs 18, leaving only 10 for the
 * name AD -- 8 characters. The default LM_BLE_DEVICE_NAME is longer than that,
 * so packing both into the advertisement made ble_gap_adv_set_fields() fail
 * with BLE_HS_EMSGSIZE and the node never advertised at all.
 *
 * The service UUID goes in the scan response, which has its own 31-byte budget,
 * leaving the advertisement for flags plus the name. A scanner still filters on
 * the UUID; it just arrives in the scan response. If a name is configured that
 * is too long even for that, it is advertised shortened rather than dropping
 * the node off the air entirely. */
#define LM_BLE_ADV_BUDGET      31u
#define LM_BLE_ADV_FLAGS_COST   3u
#define LM_BLE_ADV_HDR_COST     2u

static void advertise(void)
{
    const size_t name_budget =
        LM_BLE_ADV_BUDGET - LM_BLE_ADV_FLAGS_COST - LM_BLE_ADV_HDR_COST;
    size_t name_len = strlen(CONFIG_LM_BLE_DEVICE_NAME);
    uint8_t name_is_complete = 1;
    if (name_len > name_budget) {
        ESP_LOGW(TAG,
                 "LM_BLE_DEVICE_NAME is %u bytes; advertising the first %u "
                 "(legacy advertisement fits %u)",
                 (unsigned)name_len, (unsigned)name_budget,
                 (unsigned)name_budget);
        name_len = name_budget;
        name_is_complete = 0;
    }

    struct ble_hs_adv_fields fields = {0};
    fields.flags = BLE_HS_ADV_F_DISC_GEN | BLE_HS_ADV_F_BREDR_UNSUP;
    fields.name = (const uint8_t *)CONFIG_LM_BLE_DEVICE_NAME;
    fields.name_len = (uint8_t)name_len;
    fields.name_is_complete = name_is_complete;
    int rc = ble_gap_adv_set_fields(&fields);
    if (rc != 0) {
        ESP_LOGE(TAG, "adv fields: %d", rc);
        return;
    }

    struct ble_hs_adv_fields rsp = {0};
    rsp.uuids128 = (ble_uuid128_t *)&s_service_uuid;
    rsp.num_uuids128 = 1;
    rsp.uuids128_is_complete = 1;
    rc = ble_gap_adv_rsp_set_fields(&rsp);
    if (rc != 0) {
        /* Not fatal: the node stays discoverable by name, it just cannot be
         * filtered on the service UUID before connecting. */
        ESP_LOGW(TAG, "adv rsp fields: %d (service UUID not advertised)", rc);
    }

    struct ble_gap_adv_params params = {
        .conn_mode = BLE_GAP_CONN_MODE_UND,
        .disc_mode = BLE_GAP_DISC_MODE_GEN,
    };
    rc = ble_gap_adv_start(s_addr_type, NULL, BLE_HS_FOREVER, &params, gap_event, NULL);
    if (rc != 0) {
        ESP_LOGE(TAG, "adv start: %d", rc);
    }
}

static void on_sync(void)
{
    int rc = ble_hs_id_infer_auto(0, &s_addr_type);
    assert(rc == 0);
    advertise();
}

static void ble_host_task(void *arg)
{
    (void)arg;
    nimble_port_run();
    nimble_port_freertos_deinit();
}

static void notify_task(void *arg)
{
    (void)arg;
    lm_air_packet_t packet;
    uint8_t fragment[BLE_ATT_MTU_MAX];
    for (;;) {
        if (xQueueReceive(lm_air_radio_tx_queue(LM_AIR_LINK_BLE), &packet,
                          portMAX_DELAY) != pdTRUE) {
            continue;
        }
        if (s_conn_handle == BLE_HS_CONN_HANDLE_NONE || !s_subscribed) {
            lm_air_metric_add(LM_METRIC_QUEUE_DROP, 1);
            continue;
        }
        const uint16_t mtu = ble_att_mtu(s_conn_handle);
        const size_t capacity = mtu > 3 ? mtu - 3u : 0;
        const size_t count = lm_air_ble_fragment_count(packet.len, capacity);
        const uint16_t message_id = ++s_message_id;
        bool sent = count > 0;
        for (size_t i = 0; i < count && sent; ++i) {
            size_t fragment_len = 0;
            sent = lm_air_ble_make_fragment(&packet, message_id, (uint8_t)i,
                                            capacity, fragment, sizeof(fragment),
                                            &fragment_len) == 0;
            if (!sent) {
                break;
            }
            struct os_mbuf *om = ble_hs_mbuf_from_flat(fragment, fragment_len);
            sent = om != NULL &&
                   ble_gatts_notify_custom(s_conn_handle, s_value_handle, om) == 0;
            vTaskDelay(pdMS_TO_TICKS(5));
        }
        if (sent) {
            lm_air_metric_add(LM_METRIC_BLE_TX, 1);
        } else {
            lm_air_metric_add(LM_METRIC_QUEUE_DROP, 1);
        }
    }
}

esp_err_t lm_air_ble_start(void)
{
    lm_air_ble_reassembly_reset(&s_reassembly);
    ESP_ERROR_CHECK(nimble_port_init());
    ble_svc_gap_init();
    ble_svc_gatt_init();
    int rc = ble_svc_gap_device_name_set(CONFIG_LM_BLE_DEVICE_NAME);
    if (rc == 0) {
        rc = ble_gatts_count_cfg(s_services);
    }
    if (rc == 0) {
        rc = ble_gatts_add_svcs(s_services);
    }
    if (rc != 0) {
        return ESP_FAIL;
    }
    ble_hs_cfg.sync_cb = on_sync;
    nimble_port_freertos_init(ble_host_task);
    return xTaskCreate(notify_task, "lm_ble_tx", 4096, NULL, 5, NULL) == pdPASS
               ? ESP_OK
               : ESP_ERR_NO_MEM;
}
