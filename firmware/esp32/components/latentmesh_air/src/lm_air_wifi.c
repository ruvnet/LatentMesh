#include "lm_air_wifi.h"

#include <errno.h>
#include <fcntl.h>
#include <string.h>
#include "esp_check.h"
#include "esp_event.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "esp_wifi.h"
#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"
#include "freertos/task.h"
#include "lwip/inet.h"
#include "lwip/sockets.h"

#include "lm_air_metrics.h"
#include "lm_air_radio.h"

static const char *TAG = "lm_wifi";
static EventGroupHandle_t s_events;
static const EventBits_t CONNECTED = BIT0;
static int s_socket = -1;
static struct sockaddr_in s_peer;

static void wifi_event(void *arg,
                       esp_event_base_t base,
                       int32_t id,
                       void *data)
{
    (void)arg;
    (void)data;
    if (base == WIFI_EVENT && id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (base == WIFI_EVENT && id == WIFI_EVENT_STA_DISCONNECTED) {
        xEventGroupClearBits(s_events, CONNECTED);
        esp_wifi_connect();
    } else if (base == IP_EVENT && id == IP_EVENT_STA_GOT_IP) {
        xEventGroupSetBits(s_events, CONNECTED);
    }
}

static int open_socket(void)
{
    const int sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_IP);
    if (sock < 0) {
        return -1;
    }
    int one = 1;
    setsockopt(sock, SOL_SOCKET, SO_BROADCAST, &one, sizeof(one));
    struct sockaddr_in local = {
        .sin_family = AF_INET,
        .sin_port = htons(CONFIG_LM_WIFI_UDP_BIND_PORT),
        .sin_addr.s_addr = htonl(INADDR_ANY),
    };
    if (bind(sock, (struct sockaddr *)&local, sizeof(local)) != 0) {
        close(sock);
        return -1;
    }
    const int flags = fcntl(sock, F_GETFL, 0);
    if (flags < 0 || fcntl(sock, F_SETFL, flags | O_NONBLOCK) != 0) {
        close(sock);
        return -1;
    }
    return sock;
}

static void udp_task(void *arg)
{
    (void)arg;
    lm_air_packet_t tx;
    uint8_t rx[LM_AIR_MAX_FRAME_SIZE];
    for (;;) {
        xEventGroupWaitBits(s_events, CONNECTED, pdFALSE, pdTRUE, portMAX_DELAY);
        if (s_socket < 0) {
            s_socket = open_socket();
            if (s_socket < 0) {
                ESP_LOGE(TAG, "UDP socket: errno=%d", errno);
                vTaskDelay(pdMS_TO_TICKS(1000));
                continue;
            }
        }
        if (xQueueReceive(lm_air_radio_tx_queue(LM_AIR_LINK_WIFI), &tx,
                          pdMS_TO_TICKS(20)) == pdTRUE) {
            const ssize_t sent = sendto(s_socket, tx.data, tx.len, 0,
                                        (struct sockaddr *)&s_peer, sizeof(s_peer));
            if (sent == tx.len) {
                lm_air_metric_add(LM_METRIC_WIFI_TX, 1);
            } else {
                lm_air_metric_add(LM_METRIC_QUEUE_DROP, 1);
            }
        }
        struct sockaddr_in source;
        socklen_t source_len = sizeof(source);
        const ssize_t received = recvfrom(s_socket, rx, sizeof(rx), 0,
                                          (struct sockaddr *)&source, &source_len);
        if (received > 0) {
            if (lm_air_radio_ingest(LM_AIR_LINK_WIFI, rx, (size_t)received,
                                    LM_AIR_PAYLOAD_PUBLIC_CODEC, 0) == ESP_OK) {
                lm_air_metric_add(LM_METRIC_WIFI_RX, 1);
            }
        } else if (received < 0 && errno != EAGAIN && errno != EWOULDBLOCK) {
            ESP_LOGW(TAG, "UDP receive: errno=%d; reopening", errno);
            close(s_socket);
            s_socket = -1;
        }
    }
}

esp_err_t lm_air_wifi_start(void)
{
    if (CONFIG_LM_WIFI_SSID[0] == '\0') {
        ESP_LOGW(TAG, "%s", "Wi-Fi adapter dormant: configure LM_WIFI_SSID");
        return ESP_OK;
    }
    ESP_RETURN_ON_ERROR(esp_netif_init(), TAG, "netif init");
    esp_err_t err = esp_event_loop_create_default();
    if (err != ESP_OK && err != ESP_ERR_INVALID_STATE) {
        return err;
    }
    if (esp_netif_create_default_wifi_sta() == NULL) {
        return ESP_ERR_NO_MEM;
    }
    wifi_init_config_t init = WIFI_INIT_CONFIG_DEFAULT();
    ESP_RETURN_ON_ERROR(esp_wifi_init(&init), TAG, "Wi-Fi init");
    s_events = xEventGroupCreate();
    if (s_events == NULL) {
        return ESP_ERR_NO_MEM;
    }
    ESP_RETURN_ON_ERROR(esp_event_handler_register(WIFI_EVENT, ESP_EVENT_ANY_ID,
                                                   wifi_event, NULL),
                        TAG, "Wi-Fi event handler");
    ESP_RETURN_ON_ERROR(esp_event_handler_register(IP_EVENT, IP_EVENT_STA_GOT_IP,
                                                   wifi_event, NULL),
                        TAG, "IP event handler");
    wifi_config_t config = {0};
    strlcpy((char *)config.sta.ssid, CONFIG_LM_WIFI_SSID, sizeof(config.sta.ssid));
    strlcpy((char *)config.sta.password, CONFIG_LM_WIFI_PASSWORD,
            sizeof(config.sta.password));
    config.sta.threshold.authmode = CONFIG_LM_WIFI_PASSWORD[0] == '\0'
                                        ? WIFI_AUTH_OPEN
                                        : WIFI_AUTH_WPA2_PSK;
    ESP_RETURN_ON_ERROR(esp_wifi_set_mode(WIFI_MODE_STA), TAG, "Wi-Fi mode");
    ESP_RETURN_ON_ERROR(esp_wifi_set_config(WIFI_IF_STA, &config), TAG, "Wi-Fi config");
    ESP_RETURN_ON_ERROR(esp_wifi_start(), TAG, "Wi-Fi start");

    memset(&s_peer, 0, sizeof(s_peer));
    s_peer.sin_family = AF_INET;
    s_peer.sin_port = htons(CONFIG_LM_WIFI_UDP_PEER_PORT);
    if (inet_pton(AF_INET, CONFIG_LM_WIFI_UDP_PEER_IPV4, &s_peer.sin_addr) != 1) {
        return ESP_ERR_INVALID_ARG;
    }
    return xTaskCreate(udp_task, "lm_udp", 4096, NULL, 5, NULL) == pdPASS
               ? ESP_OK
               : ESP_ERR_NO_MEM;
}
