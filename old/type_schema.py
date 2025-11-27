"""
Entity Type Schema
==================
Defines the hierarchical type system for metadata extraction.
"""

ENTITY_TYPE_SCHEMA = """
**Person**
- Politician, Artist, Writer, Actor, Musician, Scientist, Academic, Businessperson, Athlete, ReligiousFigure, HistoricalFigure

**Location**
- Country, StateOrProvince, City, County, Region, NaturalPlace, Building, Landmark, Facility, Village, Town

**Organization**
- Company, GovernmentAgency, EducationalInstitution, ReligiousOrganization, PoliticalParty, NonProfitOrganization, MediaOrganization, FinancialInstitution, SportsTeam, CulturalInstitution

**WorkOfArt**
- Film, Book, Song, Album, Opera, Play, Painting, TelevisionSeries, Comic, VideoGame, Poem

**Event**
- PoliticalCampaign, Festival, Trial, WarOrConflict, SportsEvent, NaturalDisaster, HistoricalEvent, AwardCeremony, CulturalEvent, ScientificDiscovery

**BiologicalEntity**
- Species, Disease, Virus, Bacterium, MedicalCondition, BiologicalProcess, Anatomy

**Concept**
- PoliticalIdeology, PhilosophicalConcept, SocialSystem, ScientificTheory, LegalConcept, AdministrativeUnit, EconomicSystem, ReligiousDoctrine

**Product**
- Vehicle, Software, Device, Weapon, Brand, Infrastructure, Tool

**OrganizationCluster**
- HoldingCompany, Federation, Consortium
"""

# Structured dictionary format for programmatic access
ENTITY_TYPES = {
    "Person": [
        "Politician", "Artist", "Writer", "Actor", "Musician", 
        "Scientist", "Academic", "Businessperson", "Athlete", 
        "ReligiousFigure", "HistoricalFigure"
    ],
    "Location": [
        "Country", "StateOrProvince", "City", "County", "Region", 
        "NaturalPlace", "Building", "Landmark", "Facility", "Village", "Town"
    ],
    "Organization": [
        "Company", "GovernmentAgency", "EducationalInstitution", 
        "ReligiousOrganization", "PoliticalParty", "NonProfitOrganization", 
        "MediaOrganization", "FinancialInstitution", "SportsTeam", "CulturalInstitution"
    ],
    "WorkOfArt": [
        "Film", "Book", "Song", "Album", "Opera", "Play", 
        "Painting", "TelevisionSeries", "Comic", "VideoGame", "Poem"
    ],
    "Event": [
        "PoliticalCampaign", "Festival", "Trial", "WarOrConflict", 
        "SportsEvent", "NaturalDisaster", "HistoricalEvent", 
        "AwardCeremony", "CulturalEvent", "ScientificDiscovery"
    ],
    "BiologicalEntity": [
        "Species", "Disease", "Virus", "Bacterium", 
        "MedicalCondition", "BiologicalProcess", "Anatomy"
    ],
    "Concept": [
        "PoliticalIdeology", "PhilosophicalConcept", "SocialSystem", 
        "ScientificTheory", "LegalConcept", "AdministrativeUnit", 
        "EconomicSystem", "ReligiousDoctrine"
    ],
    "Product": [
        "Vehicle", "Software", "Device", "Weapon", "Brand", "Infrastructure", "Tool"
    ],
    "OrganizationCluster": [
        "HoldingCompany", "Federation", "Consortium"
    ]
}
