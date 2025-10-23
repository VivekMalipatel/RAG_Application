from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Optional, Any, Union

#Semantic Memory

class SemanticMemory(BaseModel):
    Subject: Optional[str] = Field(None, description="The subject of the memory")
    Predicate: Optional[str] = Field(None, description="The relationship or attribute")
    Object: Optional[str] = Field(None, description="The object or value for the predicate")
    Context: Optional[str] = Field(None, description="Supporting context for the memory")
    Importance: Optional[str] = Field(None, description="Why this memory matters")

# Profile Memory

class AddressModel(BaseModel):
    Street: Optional[str] = Field(None, description="Street address")
    City: Optional[str] = Field(None, description="City name")
    State: Optional[str] = Field(None, description="State or province")
    ZipCode: Optional[str] = Field(None, description="Postal code")
    Country: Optional[str] = Field(None, description="Country name")
    Extra: Optional[str] = Field(None, description="Additional address details")

class PhoneModel(BaseModel):
    Number: Optional[str] = Field(None, description="Phone number")
    CountryCode: Optional[str] = Field(None, description="Country dialing code")
    Extension: Optional[str] = Field(None, description="Phone extension if applicable")

class ContactBookModel(BaseModel):
    Name: Optional[str] = Field(None, description="Contact's full name")
    Pronouns: Optional[str] = Field(None, description="Contact's preferred pronouns")
    Nicknames: Optional[list[str]] = Field(None, description="Contact's nicknames")
    Relationship: Optional[str] = Field(None, description="Relationship to the user")
    DateOfBirth: Optional[str] = Field(None, description="Contact's date of birth")
    Email: Optional[dict[str,str]] = Field(None, description="Contact's email addresses keyed by identifier")
    Phone: Optional[dict[str,PhoneModel]] = Field(None, description="Contact's phone number")
    Address: Optional[dict[str, AddressModel]] = Field(None, description="Contact's address keyed by identifier")
    Notes: Optional[dict[str,str]] = Field(None, description="Additional notes about the contact keyed by identifier")
    Languages: Optional[list[str]] = Field(None, description="Languages spoken by the contact")

class PreferenceModel(BaseModel):
    Category: Optional[str] = Field(None, description="Category of the preference")
    Details: Optional[str] = Field(None, description="Specific details of the preference")
    Importance: Optional[str] = Field(None, description="Importance level of the preference")

class UserProfileMemory(BaseModel):
    PreferredName: Optional[str] = Field(None, description="Preferred form of address")
    FormalName: Optional[str] = Field(None, description="Full formal name")
    Nicknames: Optional[list[str]] = Field(None, description="Contact's nicknames")
    Pronouns: Optional[list[str]] = Field(None, description="Pronouns the user prefers")
    Locale: Optional[str] = Field(None, description="Primary locale or language")
    Timezone: Optional[str] = Field(None, description="Primary timezone")
    DateOfBirth: Optional[str] = Field(None, description="Date of birth")
    Email: Optional[dict[str,str]] = Field(None, description="Primary email address keyed by identifier")
    Phone: Optional[dict[str,PhoneModel]] = Field(None, description="Primary phone number keyed by identifier")
    Address: Optional[dict[str, AddressModel]] = Field(None, description="Address details keyed by identifier")
    Preferences: Optional[dict[str, PreferenceModel]] = Field(None, description="Durable user preferences keyed by identifier")
    Contacts: Optional[dict[str, ContactBookModel]] = Field(None, description="Important contacts keyed by identifier")
    Notes: Optional[dict[str, str]] = Field(None, description="Additional notes about the user keyed by identifier")

# Episodic Memory

class EpisodicMemoryModel(BaseModel):
    Observation: Optional[str] = Field(None, description="The observation made during the event")
    Thoughts: Optional[str] = Field(None, description="The thoughts associated with the event")
    Action: Optional[str] = Field(None, description="The action taken during the event")
    Result: Optional[str] = Field(None, description="The result of the action taken")

# Procedural Memory

class ProceduralMemoryModel(BaseModel):
    CoreDirectives: Optional[str] = Field(None, description="Primary persona and mission")
    ResponseGuidelines: Optional[str] = Field(None, description="Response structure and tone guidance")
    ToolingGuidelines: Optional[str] = Field(None, description="Expectations for tool usage")
    EscalationPolicy: Optional[str] = Field(None, description="How to handle failures or sensitive situations")
    Metadata: Optional[dict[str, Any]] = Field(None, description="Tracking metadata")


