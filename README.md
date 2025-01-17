# topological_stain_quantifier

Calculate topological map of stains using 2D images

## Use Case 1: Quantify NET in region

### Background
Neutrophils apoptose, releasing DNA that creates a dead zone. This effect is
called a neutrophil extracellular trap. In this tool, we have created a custom
measurement of the leve of NETs.

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
- Put scanning window as parameter as well
Green stain:
- low brightness (still higher than background) high nets
- Remove all granular spots and measure brightness for NET
    - Put in threshold slider for easy adjustment
- remove background (give threshold subtraction)
    - do adaptive background removal
    - Some NET regions have lower background than high-background regions
- Blue stain:
    - Use to remove blot in green
    - Does not have to be very accurate
    - Zero values in this region
    - add parameter for thresholding
