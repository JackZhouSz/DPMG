add_library(luisa-render-surfaces INTERFACE)
luisa_render_add_plugin(null CATEGORY surface SOURCES null.cpp)
luisa_render_add_plugin(matte CATEGORY surface SOURCES matte.cpp)
luisa_render_add_plugin(glass CATEGORY surface SOURCES glass.cpp)
luisa_render_add_plugin(metal CATEGORY surface SOURCES metal.cpp)
luisa_render_add_plugin(mirror CATEGORY surface SOURCES mirror.cpp)
luisa_render_add_plugin(disney CATEGORY surface SOURCES disney.cpp)
luisa_render_add_plugin(mix CATEGORY surface SOURCES mix.cpp)
luisa_render_add_plugin(layered CATEGORY surface SOURCES layered.cpp)
luisa_render_add_plugin(plastic CATEGORY surface SOURCES plastic.cpp)
luisa_render_add_plugin(substrate CATEGORY surface SOURCES plastic.cpp) # for compatibility
