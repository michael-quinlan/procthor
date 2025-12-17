"""Test the new hallway and connectivity features."""
import sys
sys.path.insert(0, '.')

from procthor.utils.types import LeafRoom, MetaRoom, PUBLIC_ROOM_TYPES, PRIVATE_ROOM_TYPES
from procthor.generation.connectivity import (
    build_room_adjacency_graph,
    validate_room_constraints,
    validate_private_room_access,
    can_reach_public_room,
)
from procthor.generation.room_specs import RoomSpec


def test_room_types():
    """Test that Hallway is a valid room type."""
    print("Testing room types...")
    
    # Hallway should be in PUBLIC_ROOM_TYPES
    assert "Hallway" in PUBLIC_ROOM_TYPES, "Hallway should be a public room type"
    
    # Private rooms should include Bedroom and Bathroom
    assert "Bedroom" in PRIVATE_ROOM_TYPES, "Bedroom should be a private room type"
    assert "Bathroom" in PRIVATE_ROOM_TYPES, "Bathroom should be a private room type"
    
    print("  ✓ Room types are correctly defined")


def test_leaf_room_constraints():
    """Test that LeafRoom accepts the new constraint parameters."""
    print("Testing LeafRoom constraints...")
    
    # Create a bedroom with constraints
    bedroom = LeafRoom(
        room_id=2,
        ratio=2,
        room_type="Bedroom",
        must_connect_to={"Hallway", "LivingRoom"},
        cannot_connect_only_to={"Bedroom", "Bathroom"},
    )
    
    assert bedroom.must_connect_to == {"Hallway", "LivingRoom"}
    assert bedroom.cannot_connect_only_to == {"Bedroom", "Bathroom"}
    
    # Create a hallway
    hallway = LeafRoom(
        room_id=3,
        ratio=1,
        room_type="Hallway",
    )
    assert hallway.room_type == "Hallway"
    
    print("  ✓ LeafRoom accepts constraint parameters")


def test_connectivity_validation():
    """Test the connectivity validation functions."""
    print("Testing connectivity validation...")
    
    # Create a simple room spec
    kitchen = LeafRoom(room_id=2, ratio=2, room_type="Kitchen")
    hallway = LeafRoom(room_id=3, ratio=1, room_type="Hallway")
    bedroom = LeafRoom(
        room_id=4, ratio=2, room_type="Bedroom",
        must_connect_to={"Hallway"},
    )
    
    spec = RoomSpec(
        room_spec_id="test",
        sampling_weight=1,
        spec=[kitchen, hallway, bedroom],
    )
    
    # Test adjacency graph building
    doors = [
        {"room0": "room|2", "room1": "room|3"},  # Kitchen-Hallway
        {"room0": "room|3", "room1": "room|4"},  # Hallway-Bedroom
    ]
    
    adjacency = build_room_adjacency_graph(doors, spec.room_type_map)
    
    assert 2 in adjacency and 3 in adjacency[2], "Kitchen should connect to Hallway"
    assert 3 in adjacency and 4 in adjacency[3], "Hallway should connect to Bedroom"
    
    # Test constraint validation - should pass
    errors = validate_room_constraints(spec, adjacency)
    assert len(errors) == 0, f"Should have no errors: {errors}"
    
    # Test with bad connectivity - bedroom only connects to another bedroom
    bad_doors = [
        {"room0": "room|2", "room1": "room|3"},  # Kitchen-Hallway
        {"room0": "room|4", "room1": "room|5"},  # Bedroom-Bedroom (bad!)
    ]
    
    bedroom2 = LeafRoom(room_id=5, ratio=2, room_type="Bedroom")
    bad_spec = RoomSpec(
        room_spec_id="bad-test",
        sampling_weight=1,
        spec=[kitchen, hallway, bedroom, bedroom2],
    )
    
    bad_adjacency = build_room_adjacency_graph(bad_doors, bad_spec.room_type_map)
    errors = validate_room_constraints(bad_spec, bad_adjacency)
    assert len(errors) > 0, "Should have errors for bad connectivity"
    
    print("  ✓ Connectivity validation works correctly")


def test_public_room_access():
    """Test that private rooms can reach public rooms."""
    print("Testing public room access...")
    
    room_type_map = {
        2: "Kitchen",
        3: "Hallway",
        4: "Bedroom",
    }
    
    # Good: Bedroom -> Hallway -> Kitchen
    good_adjacency = {
        2: {3},
        3: {2, 4},
        4: {3},
    }
    
    assert can_reach_public_room(4, good_adjacency, room_type_map), \
        "Bedroom should reach public room through hallway"
    
    # Bad: Bedroom isolated
    bad_adjacency = {
        2: {3},
        3: {2},
        4: set(),  # Isolated!
    }
    
    assert not can_reach_public_room(4, bad_adjacency, room_type_map), \
        "Isolated bedroom should not reach public room"
    
    print("  ✓ Public room access validation works correctly")


if __name__ == "__main__":
    print("=" * 50)
    print("Testing Hallway and Connectivity Features")
    print("=" * 50)
    
    test_room_types()
    test_leaf_room_constraints()
    test_connectivity_validation()
    test_public_room_access()
    
    print("=" * 50)
    print("All tests passed! ✓")
    print("=" * 50)

