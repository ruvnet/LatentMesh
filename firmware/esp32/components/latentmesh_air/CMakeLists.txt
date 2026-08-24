set(LM_AIR_SRCS
    "src/lm_air_ble_frag.c"
    "src/lm_air_crc.c"
    "src/lm_air_kiss.c"
    "src/lm_air_metrics.c"
    "src/lm_air_pipeline.c"
    "src/lm_air_policy.c"
    "src/lm_air_radio.c"
)

if(CONFIG_LM_WIFI_ENABLE)
    list(APPEND LM_AIR_SRCS "src/lm_air_wifi.c")
endif()
if(CONFIG_LM_BLE_ENABLE)
    list(APPEND LM_AIR_SRCS "src/lm_air_ble.c")
endif()
if(CONFIG_LM_UART_KISS_ENABLE)
    list(APPEND LM_AIR_SRCS "src/lm_air_kiss_uart.c")
endif()
if(CONFIG_LM_I2S_AUDIO_ENABLE)
    list(APPEND LM_AIR_SRCS "src/lm_air_i2s.c")
endif()

idf_component_register(
    SRCS ${LM_AIR_SRCS}
    INCLUDE_DIRS "include"
    REQUIRES bt c esp_driver_gpio esp_driver_i2s esp_driver_uart esp_event
             esp_netif esp_timer esp_wifi lwip
)

target_compile_options(${COMPONENT_LIB} PRIVATE -Wall -Wextra -Werror)

# The transport component includes only the public <latentmesh_air.h> header.
# ESP-IDF names the repository's c/** component `__idf_c`; a packaged component
# may instead use the more descriptive `__idf_latentmesh_c` target.
if(TARGET __idf_c)
    target_link_libraries(${COMPONENT_LIB} PUBLIC __idf_c)
    target_compile_definitions(${COMPONENT_LIB} PUBLIC LM_HAVE_PORTABLE_C=1)
elseif(TARGET __idf_latentmesh_c)
    target_link_libraries(${COMPONENT_LIB} PUBLIC __idf_latentmesh_c)
    target_compile_definitions(${COMPONENT_LIB} PUBLIC LM_HAVE_PORTABLE_C=1)
endif()
