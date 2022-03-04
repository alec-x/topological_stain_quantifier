# topological_stain_quantifier

Calculate topological map of stains using 2D images

## Use Case 1: Quantify NET in region

### Background
Neutrophils apoptose, releasing DNA that creates a dead zone. This effect is
called a neutrophil extracellular trap

### Measurements Given
**Green Stain (EGFP)** 
- Nuclear stain: Represents dead cell OR NET
    - Dead cells are very granular. NET is very cloudy

**Red Stain (Cy5)**
- All cell stain: All dead or alive cells are stained

**Blue Stain (DAPI)**
- Initial clump of NET
    - Used to start chain reaction of NET release

### Goal: Estimate NET concentration in region

- Only needs to be relatively correct. No need for exact quantities
    - More NET = higher number, less NET = lower number
Green stain:
- high brightness, high NET
    - low granularity, high NET
- Red stain:
    - normalize to topological map of red stain
        - Use gaussian/median to smooth?
