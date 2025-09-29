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

class UserProfileMemory(BaseModel):
    PreferredName: Optional[str] = Field(None, description="Preferred form of address")
    FormalName: Optional[str] = Field(None, description="Full formal name")
    Pronouns: Optional[str] = Field(None, description="Pronouns the user prefers")
    Locale: Optional[str] = Field(None, description="Primary locale or language")
    Timezone: Optional[str] = Field(None, description="Primary timezone")
    Greeting: Optional[str] = Field(None, description="Suggested greeting style")
    Summary: Optional[str] = Field(None, description="Short personalization summary")
    Email: Optional[str] = Field(None, description="Primary email address")
    Phone: Optional[str] = Field(None, description="Primary phone number")
    Address: Optional[str] = Field(None, description="Primary address")
    Preferences: Optional[str] = Field(None, description="Durable user preferences")
    Communication: Optional[str] = Field(None, description="Communication guidance")
    WorkContext: Optional[str] = Field(None, description="Work roles, teams, or projects")
    Contacts: Optional[str] = Field(None, description="Important contacts and relationships")
    Notes: Optional[str] = Field(None, description="Additional narrative context")
    Metadata: Optional[dict[str, Any]] = Field(None, description="Profile tracking metadata")

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


