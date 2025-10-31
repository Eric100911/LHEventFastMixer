"""Integration tests for LHE mixer with merge recipe functionality."""

import numpy as np
import awkward as ak
import pylhe
from LHE_mixer import lhe_event_ak_mixer
from pylhe import LHEEvent, LHEEventInfo, LHEParticle


def create_mock_lhe_event(n_particles=4, z_momentum=7000.0, offset=0):
    """Create a mock LHE event with gluons and final state particles."""
    particles = [
        LHEParticle(
            id=21,
            status=-1,
            mother1=0,
            mother2=0,
            color1=101 + offset,
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
            color2=101 + offset,
            px=0.0,
            py=0.0,
            pz=-z_momentum,
            e=abs(z_momentum),
            m=0.0,
            lifetime=0.0,
            spin=0.0,
        ),
    ]
    
    # Add final state particles
    for i in range(n_particles - 2):
        particles.append(
            LHEParticle(
                id=443,  # J/psi
                status=1,
                mother1=1,
                mother2=2,
                color1=0,
                color2=0,
                px=float(i + 1),
                py=float(i + 2),
                pz=float(i + 3),
                e=float(i + 4),
                m=3.096,
                lifetime=0.0,
                spin=0.0,
            )
        )
    
    return particles


def create_mock_awkward_source(n_events=10, n_particles=4, z_momentum=7000.0, offset=0):
    """Create a mock awkward array source for testing."""
    events = []
    for evt_idx in range(n_events):
        particles = create_mock_lhe_event(n_particles, z_momentum, offset)
        event_info = LHEEventInfo(
            nparticles=len(particles),
            pid=-1,
            weight=1.0,
            scale=91.188,
            aqed=0.007297,
            aqcd=0.118,
        )
        events.append(LHEEvent(eventinfo=event_info, particles=particles))
    
    # Convert to awkward using pylhe
    return pylhe.to_awkward(events)


def test_basic_mixing_without_merge_recipe():
    """Test basic mixing without merge recipe (original functionality)."""
    # Create three sources
    source_a = create_mock_awkward_source(n_events=20, n_particles=4, offset=0)
    source_b = create_mock_awkward_source(n_events=10, n_particles=4, offset=100)
    source_c = create_mock_awkward_source(n_events=30, n_particles=4, offset=200)
    
    sources = [source_a, source_b, source_c]
    mix_recipe = [2, 1, 3]  # 2 from A, 1 from B, 3 from C
    sources_count = [20, 10, 30]
    
    result = lhe_event_ak_mixer(
        sources=sources,
        mix_recipe=mix_recipe,
        sources_count=sources_count,
        sort_particles_by_status=False,
        merge_recipe=None,
    )
    
    # Should have 10 events (limited by min(20/2, 10/1, 30/3))
    assert len(result) == 10
    
    # Each event should have 6 sub-scatterings × 4 particles = 24 particles
    assert ak.num(result[0].particles, axis=0) == 24
    
    # Check color flow separation: colors should be offset between sources
    colors = result[0].particles.color1
    # First group should have colors around 101
    # Later groups should have much higher color values
    assert ak.max(colors) > 500  # Indicates proper color offsetting
    
    print("✓ Basic mixing without merge recipe works correctly")


def test_mixing_with_merge_recipe():
    """Test mixing with merge recipe (gluon merging)."""
    # Create three sources
    source_a = create_mock_awkward_source(n_events=20, n_particles=4, offset=0)
    source_b = create_mock_awkward_source(n_events=10, n_particles=4, offset=100)
    source_c = create_mock_awkward_source(n_events=30, n_particles=4, offset=200)
    
    sources = [source_a, source_b, source_c]
    mix_recipe = [2, 1, 3]  # 2 from A, 1 from B, 3 from C
    merge_recipe = [[1, 1, 1], [1, 0, 0], [0, 0, 2]]  # Merge some, keep some separate
    sources_count = [20, 10, 30]
    
    result = lhe_event_ak_mixer(
        sources=sources,
        mix_recipe=mix_recipe,
        sources_count=sources_count,
        sort_particles_by_status=False,
        merge_recipe=merge_recipe,
    )
    
    # Should have 10 events
    assert len(result) == 10
    
    # Each event should have particles from 3 merged groups
    # Group 1: merged (1A + 1B + 1C) = 2 gluons + 6 final = 8 particles
    # Group 2: separate (1A) = 4 particles  
    # Group 3: merged (2C) = 2 gluons + 4 final = 6 particles
    # Total: 8 + 4 + 6 = 18 particles
    expected_particles = 18
    actual_particles = ak.num(result[0].particles, axis=0)
    
    print(f"Expected particles: {expected_particles}, Actual: {actual_particles}")
    assert actual_particles == expected_particles
    
    # Check that initial state particles exist
    initial_particles = result[0].particles[result[0].particles.status == -1]
    print(f"Number of initial particles: {ak.num(initial_particles, axis=0)}")
    
    # Should have gluons from merged groups
    # Group 1 (merged): 2 gluons
    # Group 2 (not merged): 2 gluons
    # Group 3 (merged): 2 gluons
    # Total: 6 gluons
    assert ak.num(initial_particles, axis=0) == 6
    
    print("✓ Mixing with merge recipe works correctly")


def test_color_flow_preservation():
    """Test that color flow is preserved correctly during merging."""
    source_a = create_mock_awkward_source(n_events=10, n_particles=4, offset=0)
    source_b = create_mock_awkward_source(n_events=10, n_particles=4, offset=100)
    
    sources = [source_a, source_b]
    mix_recipe = [1, 1]
    merge_recipe = [[1, 1]]  # Merge both into one
    sources_count = [10, 10]
    
    result = lhe_event_ak_mixer(
        sources=sources,
        mix_recipe=mix_recipe,
        sources_count=sources_count,
        sort_particles_by_status=False,
        merge_recipe=merge_recipe,
    )
    
    # Check that color indices are valid (non-negative)
    colors1 = result[0].particles.color1
    colors2 = result[0].particles.color2
    
    assert ak.all(colors1 >= 0)
    assert ak.all(colors2 >= 0)
    
    # Check that non-zero colors are properly assigned
    non_zero_colors1 = colors1[colors1 > 0]
    non_zero_colors2 = colors2[colors2 > 0]
    
    # Colors should be in reasonable range (not too large)
    if len(non_zero_colors1) > 0:
        assert ak.max(non_zero_colors1) < 1000
    if len(non_zero_colors2) > 0:
        assert ak.max(non_zero_colors2) < 1000
    
    print("✓ Color flow preservation works correctly")


def test_mother_lineage_preservation():
    """Test that mother particle indices are preserved correctly."""
    source_a = create_mock_awkward_source(n_events=10, n_particles=4, offset=0)
    
    sources = [source_a]
    mix_recipe = [2]
    merge_recipe = [[2]]  # Merge both sub-scatterings
    sources_count = [10]
    
    result = lhe_event_ak_mixer(
        sources=sources,
        mix_recipe=mix_recipe,
        sources_count=sources_count,
        sort_particles_by_status=False,
        merge_recipe=merge_recipe,
    )
    
    # Check mother indices
    particles = result[0].particles
    mother1 = particles.mother1
    mother2 = particles.mother2
    
    # All mother indices should be valid (0 or positive)
    assert ak.all(mother1 >= 0)
    assert ak.all(mother2 >= 0)
    
    # Final state particles (status=1) should have positive mother indices
    final_particles = particles[particles.status == 1]
    assert ak.all(final_particles.mother1 > 0)
    assert ak.all(final_particles.mother2 > 0)
    
    # Mother indices should point to valid particles (within bounds)
    n_particles = ak.num(particles, axis=0)
    assert ak.all(final_particles.mother1 <= n_particles)
    assert ak.all(final_particles.mother2 <= n_particles)
    
    print("✓ Mother lineage preservation works correctly")


def test_all_merged_scenario():
    """Test scenario where all sub-scatterings are merged into one."""
    source_a = create_mock_awkward_source(n_events=10, n_particles=4, offset=0)
    source_b = create_mock_awkward_source(n_events=10, n_particles=4, offset=100)
    source_c = create_mock_awkward_source(n_events=10, n_particles=4, offset=200)
    
    sources = [source_a, source_b, source_c]
    mix_recipe = [1, 1, 1]
    merge_recipe = [[1, 1, 1]]  # Merge all three
    sources_count = [10, 10, 10]
    
    result = lhe_event_ak_mixer(
        sources=sources,
        mix_recipe=mix_recipe,
        sources_count=sources_count,
        sort_particles_by_status=False,
        merge_recipe=merge_recipe,
    )
    
    # Should have 10 events
    assert len(result) == 10
    
    # Each event should have: 2 gluons (merged) + 6 final particles = 8 particles
    assert ak.num(result[0].particles, axis=0) == 8
    
    # Should have exactly 2 initial gluons
    initial_count = ak.sum(result[0].particles.status == -1)
    assert initial_count == 2
    
    # Should have 6 final state particles
    final_count = ak.sum(result[0].particles.status == 1)
    assert final_count == 6
    
    print("✓ All merged scenario works correctly")


def test_no_merging_scenario():
    """Test scenario where no actual merging occurs (identity merge)."""
    source_a = create_mock_awkward_source(n_events=10, n_particles=4, offset=0)
    source_b = create_mock_awkward_source(n_events=10, n_particles=4, offset=100)
    
    sources = [source_a, source_b]
    mix_recipe = [1, 1]
    merge_recipe = [[1, 0], [0, 1]]  # Keep them separate
    sources_count = [10, 10]
    
    result = lhe_event_ak_mixer(
        sources=sources,
        mix_recipe=mix_recipe,
        sources_count=sources_count,
        sort_particles_by_status=False,
        merge_recipe=merge_recipe,
    )
    
    # Should have 10 events
    assert len(result) == 10
    
    # Each event should have 8 particles (2 separate sub-scatterings of 4 each)
    assert ak.num(result[0].particles, axis=0) == 8
    
    # Should have 4 initial gluons (2 per sub-scattering)
    initial_count = ak.sum(result[0].particles.status == -1)
    assert initial_count == 4
    
    print("✓ No merging scenario works correctly")


if __name__ == "__main__":
    print("Running integration tests...\n")
    
    test_basic_mixing_without_merge_recipe()
    test_mixing_with_merge_recipe()
    test_color_flow_preservation()
    test_mother_lineage_preservation()
    test_all_merged_scenario()
    test_no_merging_scenario()
    
    print("\n✓ All integration tests passed!")
