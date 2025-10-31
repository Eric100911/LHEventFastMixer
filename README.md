# LHEventFastMixer

A high-performance Python library for mixing LHE (Les Houches Event) files with support for advanced gluon merging techniques.

## Features

- **Fast Vectorized Operations**: Uses `awkward` arrays for efficient processing
- **Flexible Mixing**: Mix events from multiple sources with customizable recipes
- **Gluon Merging**: Advanced support for merging sub-scatterings with proper color flow handling
- **Color Flow Preservation**: Maintains color flow information throughout the mixing process
- **Particle Lineage Tracking**: Preserves mother-daughter relationships between particles

## Installation

```bash
pip install pylhe awkward numpy vector
```

## Basic Usage

### Simple Event Mixing

Mix events from multiple sources without gluon merging:

```python
import awkward as ak
import pylhe
from LHE_mixer import lhe_event_ak_mixer

# Load sources
source_a = pylhe.to_awkward(pylhe.read_lhe_file("source_a.lhe").events)
source_b = pylhe.to_awkward(pylhe.read_lhe_file("source_b.lhe").events)
source_c = pylhe.to_awkward(pylhe.read_lhe_file("source_c.lhe").events)

sources = [source_a, source_b, source_c]
mix_recipe = [2, 1, 3]  # Take 2 from A, 1 from B, 3 from C per output event
sources_count = [100, 50, 150]  # Use first N events from each source

# Mix without gluon merging
result = lhe_event_ak_mixer(
    sources=sources,
    mix_recipe=mix_recipe,
    sources_count=sources_count,
    sort_particles_by_status=False,
    merge_recipe=None  # No gluon merging
)
```

### Advanced: Gluon Merging

The merge recipe feature allows you to control how sub-scatterings are merged together, enabling the production of events like J/psi+J/psi+Y(1S) from individual J/psi and Y(1S) events.

#### Understanding Mix Recipe vs Merge Recipe

- **Mix Recipe**: `[2, 1, 3]` means take 2 events from source A, 1 from B, and 3 from C
- **Merge Recipe**: `[[1, 1, 1], [1, 0, 0], [0, 0, 2]]` specifies how to merge these sub-scatterings

Each row in the merge recipe represents one merged sub-scattering in the output:
- Row 1: `[1, 1, 1]` - Merge 1 from A, 1 from B, 1 from C together (gluon merging)
- Row 2: `[1, 0, 0]` - Keep 1 from A separate (no merging)
- Row 3: `[0, 0, 2]` - Merge 2 from C together (gluon merging)

The sum of each column must equal the corresponding value in the mix recipe.

#### Example: J/psi + J/psi + Y(1S) Production

```python
# For producing J/psi + J/psi + Y(1S) events
# Source A: J/psi events
# Source B: J/psi events  
# Source C: Y(1S) events

sources = [jpsi_source, jpsi_source, y1s_source]
mix_recipe = [1, 1, 1]  # Take 1 from each

# Option 1: Merge all three into one event (triple production)
merge_recipe = [[1, 1, 1]]

# Option 2: Two separate J/psi, one Y(1S)
merge_recipe = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

# Option 3: Merge two J/psi, keep Y(1S) separate
merge_recipe = [[1, 1, 0], [0, 0, 1]]

result = lhe_event_ak_mixer(
    sources=sources,
    mix_recipe=mix_recipe,
    sources_count=[100, 100, 100],
    merge_recipe=merge_recipe  # Enable gluon merging
)
```

## Merge Recipe Validation

The library automatically validates merge recipes to ensure compatibility:

```python
from LHE_mixer import validate_merge_recipe

mix_recipe = [2, 1, 3]
merge_recipe = [[1, 1, 1], [1, 0, 0], [0, 0, 2]]

# Validates that:
# - Each row has the same length as mix_recipe
# - Sum of each column equals the corresponding mix_recipe value
# - All values are non-negative integers
validate_merge_recipe(mix_recipe, merge_recipe)  # Returns True or raises ValueError
```

## Gluon Merging Technique

When merge recipes are used, the library implements an advanced gluon merging technique:

1. **Initial State Merging**: Multiple initial gluons are merged into a single pair
2. **Momentum Conservation**: Total momentum is preserved during merging
3. **Color Flow Management**: Color indices are properly reassigned to maintain color flow
4. **Final State Handling**: Final state particles are connected to the merged gluons

### Technical Details

The gluon merging process:
- Calculates total beam momentum from all initial gluons
- Creates two new gluons with momenta that conserve the total 4-momentum
- Reassigns color flow indices to avoid conflicts
- Updates mother particle indices to point to the new gluons

## Color Flow and Lineage Preservation

The library carefully preserves two critical pieces of information:

### Color Flow
- Color indices are offset between sources to prevent conflicts
- Non-zero colors are preserved and properly incremented
- Zero colors (for non-colored particles) remain zero

### Particle Lineage
- Mother particle indices (`mother1`, `mother2`) are updated to account for particle position changes
- Lineage is maintained across merged events
- Invalid mother indices (0 or negative) are preserved

## API Reference

### `lhe_event_ak_mixer`

Main function for mixing LHE events.

**Parameters:**
- `sources` (List[ak.Array]): List of awkward arrays containing LHE events
- `mix_recipe` (List[int]): Number of events to take from each source
- `sources_count` (Optional[List[int]]): Maximum events to use from each source
- `sort_particles_by_status` (bool): Whether to sort particles by status
- `merge_recipe` (Optional[List[List[int]]]): Merge recipe for gluon merging

**Returns:**
- `ak.Array`: Mixed events in LHE awkward array format

### `validate_merge_recipe`

Validates merge recipe compatibility with mix recipe.

**Parameters:**
- `mix_recipe` (List[int]): The mix recipe
- `merge_recipe` (List[List[int]]): The merge recipe to validate

**Returns:**
- `bool`: True if valid

**Raises:**
- `ValueError`: If merge recipe is invalid or incompatible

### `lhe_merge_gluons_ak`

Low-level function for gluon merging (typically not called directly).

**Parameters:**
- `particles` (ak.Array): Particles organized as [event, sub_scattering, particle]
- `merge_groups` (List[List[int]]): Sub-scattering indices to merge together

**Returns:**
- `ak.Array`: Merged particles with updated gluons and color flow

## Examples

### Example 1: Double Parton Scattering

```python
# Produce J/psi + J/psi events from single J/psi events
jpsi_events = pylhe.to_awkward(pylhe.read_lhe_file("jpsi.lhe").events)

result = lhe_event_ak_mixer(
    sources=[jpsi_events, jpsi_events],
    mix_recipe=[1, 1],
    sources_count=[1000, 1000],
    merge_recipe=[[1, 1]]  # Merge both J/psi into one event
)
```

### Example 2: Triple Production with Partial Merging

```python
# A + A + B production with first two A's merged
source_a = pylhe.to_awkward(pylhe.read_lhe_file("particle_a.lhe").events)
source_b = pylhe.to_awkward(pylhe.read_lhe_file("particle_b.lhe").events)

result = lhe_event_ak_mixer(
    sources=[source_a, source_b],
    mix_recipe=[2, 1],  # 2 A's and 1 B
    sources_count=[1000, 500],
    merge_recipe=[
        [2, 0],  # Merge both A's together
        [0, 1]   # Keep B separate
    ]
)
```

### Example 3: No Merging (Identity)

```python
# Mix events but keep all sub-scatterings separate
result = lhe_event_ak_mixer(
    sources=[source_a, source_b],
    mix_recipe=[1, 1],
    sources_count=[1000, 1000],
    merge_recipe=[
        [1, 0],  # First A separate
        [0, 1]   # B separate
    ]
)
# This produces the same result as merge_recipe=None for this case
```

## Writing Output

Convert the result back to LHE format:

```python
from LHE_mixer import lhe_ak_to_lhe_file

# Get init info from one of the sources
init_info = pylhe.read_lhe_init("source_a.lhe")

# Create LHE file
lhe_file = lhe_ak_to_lhe_file(result, init_info)

# Write to file
pylhe.write_lhe_file("output.lhe", lhe_file)
```

## Testing

Run the test suite:

```bash
# Run all tests
pytest test_merge_recipe.py test_integration.py -v

# Run specific test file
pytest test_merge_recipe.py -v

# Run with coverage
pytest --cov=LHE_mixer test_merge_recipe.py test_integration.py
```

## Performance Considerations

- The library uses vectorized operations with `awkward` arrays for efficiency
- Memory usage scales with the number of events and particles
- For very large files, consider processing in batches
- Gluon merging adds computational overhead but is still efficient

## License

[Add your license here]

## References

- Based on techniques from the OniaEventMixer project
- Uses the [pylhe](https://github.com/scikit-hep/pylhe) library for LHE file handling
- Leverages [awkward-array](https://awkward-array.org/) for efficient array operations

## Contributing

Contributions are welcome! Please ensure:
- All tests pass
- New features include tests
- Code follows the existing style
- Documentation is updated

## Citation

If you use this library in your research, please cite:

```
[Add citation information]
```
