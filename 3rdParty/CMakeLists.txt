# ------------------------------------------------------------------
# uniengine
# ------------------------------------------------------------------
add_subdirectory(UniEngine-dev/ EXCLUDE_FROM_ALL)
set(UNIENGINE_INCLUDES ${UNIENGINE_INCLUDES} PARENT_SCOPE)
set(UNIENGINE_PCH ${UNIENGINE_PCH} PARENT_SCOPE)