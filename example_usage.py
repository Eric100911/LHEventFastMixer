#!/usr/bin/env python3
"""
Example usage of LHEventFastMixer with merge recipe functionality.

This script demonstrates how to:
1. Create mock LHE events for testing
2. Mix events with different merge recipes
3. Validate merge recipes
4. Write output to LHE files
"""

import pylhe
import awkward as ak
from LHE_mixer import (
    lhe_event_ak_mixer,
    validate_merge_recipe,
    lhe_ak_to_lhe_file,
)
from pylhe import LHEEvent, LHEEventInfo, LHEParticle


def create_sample_event(particle_id=443, z_momentum=7000.0, event_number=0):
    """Create a sample LHE event with two initial gluons and a final state particle."""
    particles = [
        LHEParticle(
            id=21,
            status=-1,
            mother1=0,
            mother2=0,
            color1=101,
            color2=0,
            px=0.0,
            py=0.0,
            pz=z_momentum,
            e=abs(z_momentum),
            m=0.0,
            lifetime=0.0,
            spin=0.0,
        ),
        LHEParticle(
            id=21,
            status=-1,
            mother1=0,
            mother2=0,
            color1=0,
            color2=101,
            px=0.0,
            py=0.0,
            pz=-z_momentum,
            e=abs(z_momentum),
            m=0.0,
            lifetime=0.0,
            spin=0.0,
        ),
        LHEParticle(
            id=particle_id,
            status=1,
            mother1=1,
            mother2=2,
            color1=0,
            color2=0,
            px=1.0 + event_number * 0.1,
            py=2.0 + event_number * 0.1,
            pz=3.0 + event_number * 0.1,
            e=10.0 + event_number * 0.1,
            m=3.096 if particle_id == 443 else 9.460,
            lifetime=0.0,
            spin=0.0,
        ),
    ]

    event_info = LHEEventInfo(
        nparticles=len(particles),
        pid=-1,
        weight=1.0,
        scale=91.188,
        aqed=0.007297,
        aqcd=0.118,
    )

    return LHEEvent(eventinfo=event_info, particles=particles)


def example_1_basic_mixing():
    """Example 1: Basic mixing without merge recipe."""
    print("=" * 60)
    print("Example 1: Basic Mixing (No Gluon Merging)")
    print("=" * 60)

    # Create mock sources: J/psi events
    print("Creating 20 J/psi events for each source...")
    jpsi_events_a = [create_sample_event(443, 7000.0, i) for i in range(20)]
    jpsi_events_b = [create_sample_event(443, 6500.0, i) for i in range(20)]

    # Convert to awkward arrays
    source_a = pylhe.to_awkward(jpsi_events_a)
    source_b = pylhe.to_awkward(jpsi_events_b)

    # Mix: Take 1 from A and 1 from B per output event
    print("\nMixing with recipe [1, 1] (no gluon merging)...")
    result = lhe_event_ak_mixer(
        sources=[source_a, source_b],
        mix_recipe=[1, 1],
        sources_count=[20, 20],
        merge_recipe=None,  # No gluon merging
    )

    print(f"✓ Created {len(result)} mixed events")
    print(f"✓ Each event has {ak.num(result[0].particles, axis=0)} particles")
    print(
        f"  (2 sub-scatterings × 3 particles each = 6 particles per event)"
    )


def example_2_gluon_merging():
    """Example 2: Gluon merging - merge two J/psi into one event."""
    print("\n" + "=" * 60)
    print("Example 2: Gluon Merging - Double J/psi Production")
    print("=" * 60)

    # Create mock sources
    print("Creating 20 J/psi events for each source...")
    jpsi_events_a = [create_sample_event(443, 7000.0, i) for i in range(20)]
    jpsi_events_b = [create_sample_event(443, 6500.0, i) for i in range(20)]

    source_a = pylhe.to_awkward(jpsi_events_a)
    source_b = pylhe.to_awkward(jpsi_events_b)

    # Mix: Take 1 from A and 1 from B, then MERGE them
    mix_recipe = [1, 1]
    merge_recipe = [[1, 1]]  # Merge both into one event

    print("\nValidating merge recipe...")
    validate_merge_recipe(mix_recipe, merge_recipe)
    print("✓ Merge recipe is valid")

    print("\nMixing with gluon merging...")
    result = lhe_event_ak_mixer(
        sources=[source_a, source_b],
        mix_recipe=mix_recipe,
        sources_count=[20, 20],
        merge_recipe=merge_recipe,
    )

    print(f"✓ Created {len(result)} mixed events")
    print(f"✓ Each event has {ak.num(result[0].particles, axis=0)} particles")
    print(
        f"  (2 initial gluons + 2 J/psi particles = 4 particles per event)"
    )


def example_3_complex_merging():
    """Example 3: Complex merging - J/psi + J/psi + Y(1S) with partial merging."""
    print("\n" + "=" * 60)
    print("Example 3: Complex Merging - J/psi + J/psi + Y(1S)")
    print("=" * 60)

    # Create three sources: J/psi, J/psi, Y(1S)
    print("Creating events...")
    jpsi_events_1 = [create_sample_event(443, 7000.0, i) for i in range(30)]
    jpsi_events_2 = [create_sample_event(443, 6800.0, i) for i in range(30)]
    upsilon_events = [create_sample_event(553, 6500.0, i) for i in range(30)]

    source_jpsi1 = pylhe.to_awkward(jpsi_events_1)
    source_jpsi2 = pylhe.to_awkward(jpsi_events_2)
    source_y1s = pylhe.to_awkward(upsilon_events)

    # Mix recipe: 2 J/psi from source 1, 1 J/psi from source 2, 1 Y(1S) from source 3
    # Total: 4 sub-scatterings per output event
    mix_recipe = [2, 1, 1]

    # Merge recipe:
    # - Merge first J/psi with second J/psi (from source 1 and 2)
    # - Keep second J/psi from source 1 separate
    # - Keep Y(1S) separate
    merge_recipe = [
        [1, 1, 0],  # Merge 1 from source 1, 1 from source 2
        [1, 0, 0],  # Keep 1 from source 1 separate
        [0, 0, 1],  # Keep Y(1S) separate
    ]

    print(f"\nMix recipe: {mix_recipe}")
    print(f"Merge recipe: {merge_recipe}")
    print("\nValidating merge recipe...")
    validate_merge_recipe(mix_recipe, merge_recipe)
    print("✓ Merge recipe is valid")

    print("\nMixing with complex merge recipe...")
    result = lhe_event_ak_mixer(
        sources=[source_jpsi1, source_jpsi2, source_y1s],
        mix_recipe=mix_recipe,
        sources_count=[30, 30, 30],
        merge_recipe=merge_recipe,
    )

    print(f"✓ Created {len(result)} mixed events")
    print(f"✓ Each event has {ak.num(result[0].particles, axis=0)} particles")
    print("  Breakdown:")
    print("  - 1st merged group (J/psi + J/psi): 2 gluons + 2 final = 4 particles")
    print("  - 2nd separate group (J/psi): 3 particles")
    print("  - 3rd separate group (Y(1S)): 3 particles")
    print("  - Total: 4 + 3 + 3 = 10 particles per event")


def example_4_validation():
    """Example 4: Demonstrate merge recipe validation."""
    print("\n" + "=" * 60)
    print("Example 4: Merge Recipe Validation")
    print("=" * 60)

    # Valid recipe
    print("\n1. Valid merge recipe:")
    mix_recipe = [2, 1, 3]
    merge_recipe = [[1, 1, 1], [1, 0, 0], [0, 0, 2]]
    print(f"   Mix recipe: {mix_recipe}")
    print(f"   Merge recipe: {merge_recipe}")
    try:
        validate_merge_recipe(mix_recipe, merge_recipe)
        print("   ✓ Valid!")
    except ValueError as e:
        print(f"   ✗ Invalid: {e}")

    # Invalid recipe - wrong sum
    print("\n2. Invalid merge recipe (wrong column sum):")
    mix_recipe = [2, 1, 3]
    merge_recipe = [[1, 1, 1], [1, 0, 0], [1, 0, 2]]  # First column sums to 3, not 2
    print(f"   Mix recipe: {mix_recipe}")
    print(f"   Merge recipe: {merge_recipe}")
    try:
        validate_merge_recipe(mix_recipe, merge_recipe)
        print("   ✓ Valid!")
    except ValueError as e:
        print(f"   ✗ Invalid: {e}")

    # Invalid recipe - negative values
    print("\n3. Invalid merge recipe (negative values):")
    mix_recipe = [2, 1, 3]
    merge_recipe = [[1, 1, 1], [1, 0, -1], [0, 0, 3]]
    print(f"   Mix recipe: {mix_recipe}")
    print(f"   Merge recipe: {merge_recipe}")
    try:
        validate_merge_recipe(mix_recipe, merge_recipe)
        print("   ✓ Valid!")
    except ValueError as e:
        print(f"   ✗ Invalid: {e}")


def main():
    """Run all examples."""
    print("\n")
    print("*" * 60)
    print("* LHEventFastMixer - Merge Recipe Examples")
    print("*" * 60)

    example_1_basic_mixing()
    example_2_gluon_merging()
    example_3_complex_merging()
    example_4_validation()

    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
