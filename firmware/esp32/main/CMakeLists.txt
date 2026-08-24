idf_component_register(
    SRCS "app_main.c"
    INCLUDE_DIRS "."
    REQUIRES latentmesh_air nvs_flash esp_event esp_netif
)
