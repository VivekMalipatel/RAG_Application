## Precontext


Out main goal is to provide a prper UI for the Indexer API and Agent API that I have built as of now.

Our first milestone will modifying this AgentchatUI like how we want but still keeping it as subtree so that we need import all the new releases fromt he Repo

The topmost goals are,

Sticly adhere to our current abstraction principles while adhering to langgraph native implemenration and integrations so that our coding effort is minimal. SO inetad of going for this custom built chat completions api at AgentAPI/app/api we are using AgentAPI/app/langgraph_app as suggested by the forked langgrah docs at AgentChatUI/docs (this fodler is not our docs but we forked langgraph docs to refer)

As of now the basic fucntionality we need is

1. User should be able to Login to the UI
2. Role Hierarchy : Admin, User
3. Admins should have a new page that can add new users and assign roles to them
3. User should be able to upload their docuemtns into this UI (that will call the indexer API to index them) (We need pretty sophesticalted fucntionalities here that we will discuss later)
4. User should be able to choose between agents (here we have two types of agents Util Agents -> helper agents that are basically toggles in the UI and Acutal Agent -> Actual agents that are behaviorally and graphically deisgned to work in a certial way also ay utilize util agents liek chat agent deep researhc egnt etc). As of now we will just focus oc hcat agent but we need to builtd out thing in such a way that we have a proper framework and abstraction to easilty build new agents and add it to the list. This agnet selection should show up as how we select model in other chat UIs, as the knowlesge search and all the toggles will be like adding capabiltites (these toggles should be shown based on the agent's capabilties and otpion, liek chat agent can have knowledge search toggle but deep research cannot have this as it adapts this by default so out system should be abstracted in such a way that we can easily configure these things for UI). We now render the agent picker and capability toggles directly from the catalog metadata so future agents only need catalog updates.



Additonal things to be done were:

1. Out current Agent API has a robust abstraction but lacks native langgraph suggested integrations and implemetation, as it was built thinking that we will build custom APIs for everything but looks like langraph has already done all the work for us. and we just need to adapt it propelry (There is a UI that is still working in prod so we need to maitintg that legacy fucntionality until htis new one is copletely build and we can clean all the old shit after that)
2. Thsi New Agent UI enables the crazy fucntionality of human in the loop, checkpoint eidt and time travel. SO When I was building this agent API before I thought it as a later thing and didn't event thing about implementing these stuff propelry. So we need to proeprly bake this into out abstractions wherever required


The piority will always be we entering into langgraph implementation to build all our custom logic so that we adhere to langraph framework standards and recommendations 

all the Uvicron servers and stuff will be made legacy once this new UI taht we ar ebuilding is done. but as of now we will not touch the old stuff


So this is the basic pan in my mind and there is not end of possibiltiis we willbe adding here to the UI and agent. We will be doing everyting step by step and we get new ideas we will implement them. So our coding standards and abstractions should be built in such a way that we alawys follow langgraph principles and out code should be properly tought so that ist extremely extensible in terms of both UI and Agent API (aslo indexer API)

My Backgorund:

You can cnsider me a beast in Backend development but I have literally no idea and exp in frontend. i am doing this because I need some UI to expose all the backend. So you need ot guilde me like a kid here.

Core thing is:

whatever we do here, you soudl always be in context about all the things. Before you do or decide anythign you should do your complete regressipna analysis with teh code base and mainly AgentChatUI/docs so that we donot get out of our track. ALso stricly adhere to coding instructions

## Context
We are building a comprehensive interface for the AgentAPI and IndexerAPI that already power our backend systems. The Agent Chat UI is pulled in as a git subtree so we can stay close to upstream while layering our own features. The long-term roadmap includes countless enhancements—knowledge search toggles, deep research flows, human-in-the-loop actions—so we must keep the architecture disciplined from day one. Legacy OpenAI-style APIs must remain functional until this new UI is complete. I am extremely comfortable with backend work but new to frontend development, so every step should be guided and grounded in the LangGraph documentation mirrored under `AgentChatUI/docs`.

## Vision
Deliver a production-grade UI that sits on top of our AgentAPI and IndexerAPI while embracing LangGraph-native deployment patterns. The UI must remain in sync with the upstream `agent-chat-ui` project (managed as a git subtree) so future updates can be pulled with minimal conflict.

## Guiding Principles
- Respect existing backend abstractions (BaseAgent, util agents, Redis memories) while exposing them through `AgentAPI/app/langgraph_app` instead of bespoke REST shims.
- Follow LangGraph documentation in `AgentChatUI/docs` for application structure, deployment, and subgraph composition. Never implement a feature without double-checking those references.
- Keep the subtree clean: local customizations should layer on top of upstream patterns and be easy to rebase.
- Build extensibility in from the start—every new agent or capability must plug into a shared config schema that the UI can read dynamically.

## Current Stack Snapshot
- **Backend**: `AgentAPI/app/langgraph_app` runs the LangGraph app (chat agent with knowledge search subgraph) pinned to LangGraph 0.6.11 with FastAPI 0.115.14 so CopilotKit 0.1.69 stays compatible and branch navigation keeps working. Legacy OpenAI-style FastAPI routes remain for the existing production UI.
- **Indexer**: separate API handles document ingestion; UI will interact with it via authenticated endpoints.
- **Frontend**: `AgentChatUI/agent-chat-ui` (Next.js) now layers our catalog-driven agent selector (with icons) and capability toggle menu on top of the upstream chat, stays pointed at our LangGraph instance with org/user/thread IDs, and includes a toolbar toggle for the Stream Inspector to capture raw SSE events.
- **Docs**: LangGraph + LangChain documentation mirrored under `AgentChatUI/docs` for local reference.

## Milestones
1. **Authentication & Roles**
	- Implement login (username/password or SSO placeholder) and issue session tokens.
	- Support role hierarchy (Admin, User) with authorization guards in both UI routes and backend endpoints.
	- Add an Admin dashboard for user management (create users, assign roles, reset credentials).
	- Stand up `AgentAPI/backend` with async SQLAlchemy (asyncpg) pointed at the shared Postgres instance; expose session helpers for API + LangGraph usage.

2. **Document Upload & Indexer Integration**
	- Design the UI workflow for uploading documents (queueing, status, error handling).
	- Call IndexerAPI for ingestion; surface progress and history per user/org.
	- Plan for advanced features (metadata tagging, delete/reindex) but focus on core upload first.

3. **Agent & Capability Selection**
    - Define a backend metadata contract describing each agent’s capabilities and default util toggles. **Status: contract exposed at `/agents/catalog`, icons resolved in catalog provider.**
    - Render agent selection similar to model pickers; show capability toggles based on the chosen agent. **Status: complete — dropdown mirrors ChatGPT model picker, single-capability agents render an inline switch.**
    - Ensure knowledge search and other util agents plug in as LangGraph subgraphs or tool nodes, not ad-hoc HTTP tools. **Status: ongoing — frontend wiring done, backend enforcement next.**

4. **LangGraph-Native Enhancements**
	- Embrace checkpoints, human-in-the-loop patches, and time-travel APIs per LangGraph docs.
	- Keep legacy FastAPI routes functional until the new UI fully replaces them.
	- Document shared abstractions so future agents slot in with minimal work.

## Operating Rules
- Before coding, re-read relevant sections in `AgentChatUI/docs`; keep regressions in check by understanding dependencies and existing behavior.
- Maintain ASCII-only edits unless files already use other characters.
- Keep code comments minimal—only add TODOs or clarifying notes for non-obvious logic.
- Keep FastAPI <0.116 until CopilotKit ships an updated range; re-evaluate the pin once upstream loosens the requirement.
- Respect the subtree workflow; no unnecessary file churn.

## Immediate Next Actions
1. Capture representative runs with the Stream Inspector and catalog the event payload shapes so we know exactly where to attach org/user role metadata. **Status: complete** — see "Stream Event Payload Notes and ex:" below.
2. Wire the AgentChatUI frontend to the new auth endpoints (login, logout, me) and persist tokens/roles client-side. **Status: complete** — session provider drives the layout, chat runs with bearer headers, and auth mutations clear session state reliably.
3. Design the admin-facing UI workflow for user management (list, create, assign roles) using the new backend routes. **Status: complete — admin page live with create/update flows.**
4. Define the matching TypeScript types and client utilities for the agent capability schema exposed at `/agents/catalog`. **Status: complete — catalog provider decorates agents/capabilities with icons.**
5. Align backend util-agent execution with the new capability toggles (ensure enabling/disabling flags propagates into graph config). **Status: in progress — stream submits now forward `enable_knowledge_search`; confirm LangGraph gating after next backend deploy.**
6. Add catalog validation so every capability and agent registers an icon (fallback list lives in `src/lib/catalog-icons.ts`). **Status: in progress — backend now enforces icon metadata; extend lint when adding more agents.**
7. Propagate session metadata: prefer the authenticated user id (and eventually org memberships) when issuing LangGraph requests so threads stay scoped per account. **Status: in progress — Stream/Thread providers now consume the Auth context; org modelling queued.**
8. Draft the organisation & workspace model (multi-tenant, many-to-many user memberships) and bake the admin UX for assigning roles/orgs. **Status: pending — see “Organisation Management Plan” below.**
9. Relayout the UI controls so the account menu sits bottom-left with Settings (admin only) and Sign out, plus placeholders for future chat deletion. **Status: in progress — menu shell landed; thread purge wiring intentionally deferred.**
10. Plan safe thread deletion/cleanup that also purges LangGraph checkpoints. **Status: deferred — research LangGraph APIs before implementation.**
11. Migrate legacy user/org metadata from the Neo4j store into the new Postgres-backed auth tables (reuse existing UUIDs where possible, issue fresh credentials). **Status: in progress — distinct `user_id` values collected from Neo4j (255e34cc-3de9-4475-8dc1-a928084c398d, 8606867c-cc84-4a91-982e-1af49e6608aa, e3f5db54-2fad-491e-85a1-75ea12c23c3c) with `org_id` 1; awaiting manual email/name mapping confirmation.**

- **Catalog scope trim**: update the backend `agents/catalog` response (and corresponding types) so the only advertised agent is the chat agent and the only capability listed is knowledge search. **Status: stable — current response already matches this scope.**
- **Environment + health checks**: add the required `.env` in agent-chat-ui, double-check backend ENV/DB connectivity, run the backend, confirm APIs respond, then bounce the frontend and verify end-to-end. **Status: ongoing — pip install now clean after aligning FastAPI 0.115.14, rerun backend/Frontend smoke tests after the next deploy.**
- **Database bootstrap**: connect to Postgres using creds from .env, create migrations/tables as needed, and insert a default admin user so login works immediately. **Status: complete — admin seed verified.**

## Organisation Management Plan
- **Data model**: introduce an `organisations` table (`id`, `name`, `slug`, `created_at`, `updated_at`) plus a join table `organisation_members` with `(organisation_id, user_id, roles[], is_default)` so users can belong to multiple organisations and maintain per-org roles. Tie future agent runs to `(org_id, user_id)` pairs.
- **Role scoping**: keep global roles (`admin`) for platform-level operations, but add per-organisation roles (owner, manager, member) stored in the membership join row. Backend guards must check both global admin and org-scoped permissions before allowing management actions.
- **API surface**: new admin endpoints for creating organisations, listing memberships, inviting/assigning users, and setting defaults. Expose `GET /organisations`, `POST /organisations`, `POST /organisations/{id}/members`, `DELETE /organisations/{id}/members/{user_id}`, plus helpers to switch the active org in the UI session.
- **UI workflow**: extend the admin console with tabs for Organisations and Members. Admins can create an org, add/remove members, elevate to org owner/manager, and view pending invitations. Regular users get a lightweight org switcher in the footer menu once multiple orgs exist.
- **Thread scoping**: update thread search metadata (`org_id`, `user_id`) once the schema lands so history stays partitioned per organisation. Guard ingestion/upload endpoints with the same scoping to avoid cross-org leakage.
- **Open questions**: design onboarding defaults (auto-create personal org? assign to shared demo org?), invitation email flows, and how to surface org context inside LangGraph runs for audit trails.

## Legacy Data Migration Plan
- **Scope**: preserve existing Neo4j-ingested content and associated UUIDs so long-time users retain history after the move to the Postgres-backed auth system.
- **Identifiers**: Neo4j currently references three distinct `user_id` values (`255e34cc-3de9-4475-8dc1-a928084c398d`, `8606867c-cc84-4a91-982e-1af49e6608aa`, `e3f5db54-2fad-491e-85a1-75ea12c23c3c`) under a shared `org_id` of `1`; Postgres seeding must respect these IDs while issuing new usernames/passwords.
- **User profile mapping**: user provided JSON (7 total accounts) supplies names, emails, and roles; store these in a staging sheet and seed corresponding records into `users`, `user_credentials`, and forthcoming `organisation_members` tables with admin roles by default unless specified otherwise.
- **Passwords**: generate temporary random passwords (force reset on first login) or route through the existing invite flow; ensure hashed storage meets backend constraints (bcrypt via `passlib`).
- **Document ownership**: once users are seeded, backfill LangGraph thread metadata and Indexer records with the legacy `user_id`/`org_id` combos so uploads continue to resolve.
- **Verification**: after migration, run smoke tests—log in with migrated accounts, verify document listings, confirm chat history attaches to the expected user/org, and ensure Neo4j data remains untouched until cutover is complete.

## Stream Event Payload Notes
- **metadata**: `{ run_id, attempt }` seeds each run prior to graph execution.
- **langchain**: carries callback info `{ event, data, name, run_id, metadata, parent_ids }`; metadata already includes org_id, user_id, assistant_id, thread_id, checkpoint ids, and auth context flags.
- **checkpoints**: exposes persisted state `{ config, parent_config, values, metadata, next, tasks }` for the active checkpoint, mirroring LangGraph loop/plan/chat phases.
- **debug**: mirrors checkpoint payloads with `step`, `timestamp`, `type`, and `payload`, useful for auditing state transitions.
- **updates**: streams plan snapshots `{ plan }`, reflecting pending messages, token usage, next action, and capability toggles.
- **tasks**: reports node execution `{ id, name, error, result, interrupts }`; result embeds the same message stack and usage metadata as checkpoints.
- **finish**: final envelope with `{ values, next: [], tasks: [], metadata, checkpoint lineage, interrupts }` plus the top-level `{ run_id, thread_id }` block for correlation.

## Auth API Contract
- `POST /auth/register` body `{ email, password, full_name? }` → `UserRead` with default role `user`.
- `POST /auth/login` body `{ email, password }` → `{ token, expires_at, user }`; client stores bearer token and expiry.
- `GET /auth/me` header `Authorization: Bearer <token>` → `UserRead` for active session.
- `POST /auth/logout` header bearer token → `204`; session revoked server-side.
- `GET /admin/users` admin bearer token → `UserRead[]` with role lists.
- `POST /admin/users` admin bearer token body `{ email, password, full_name?, roles[] }` → `UserRead` with assigned roles.
- `PUT /admin/users/{user_id}/roles` admin bearer token body `{ roles[] }` → updated `UserRead`; empty roles defaults to `user`.
- Tokens are opaque, SHA-256 hashed in storage, TTL set by `BACKEND_SESSION_TTL_SECONDS` (default 86400s); UI should handle expiry/refresh accordingly.

## Backend Architecture (WIP)
- New module: `AgentAPI/backend` for web auth, role management, and Indexer orchestration. Keeps baggage out of LangGraph app but lives in same repo for tight coupling.
- Database: shared Postgres at `localhost:5432` (`postgres` user, password provided in secure env). Reuse the async SQLAlchemy pattern from `spectra-sight-service/core/database_client.py` (create_async_engine, async_sessionmaker, contextmanager helpers, connection test, cleanup hooks).
- Dependencies: use asyncpg driver, SQLAlchemy 2.x. Keep configuration in env vars (e.g., `BACKEND_DB_HOST`, `BACKEND_DB_USER`, etc.) and expose via Pydantic settings object.
- Integration: LangGraph nodes can call into backend services directly when needed (no extra microservice boundary). Legacy FastAPI routes can migrate gradually onto the shared session helpers.

Use this document as the living plan—update it whenever scope, priorities, or architecture decisions shift. It keeps us aligned even after long sessions.