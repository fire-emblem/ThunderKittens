#pragma once

namespace bf16_c500_tk_cute_local::cute_tk {

struct layoutc_semantic_tag {};
struct swizzled_tn_semantic_tag {};
struct continuousc_semantic_tag {};
struct continuousc_reusea_semantic_tag {};
struct continuousc_reusea_layoutc_semantic_tag {};
struct square_tt_semantic_tag {};

template <typename SemanticTag, typename TileShape, typename GeometryAtom,
          typename SchedulePolicy, typename StageLayoutAtom>
struct family_pattern {
    using semantic_tag = SemanticTag;
    using tile_shape = TileShape;
    using geometry_atom = GeometryAtom;
    using geometry_provider = typename geometry_atom::provider;
    using host_layout = typename geometry_atom::host_layout;
    using schedule_policy = SchedulePolicy;
    using stage_layout_atom = StageLayoutAtom;
};

} // namespace bf16_c500_tk_cute_local::cute_tk
