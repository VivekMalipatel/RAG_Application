"use client";

import { useEffect, useState, useCallback } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { toast } from "sonner";
import { useAuth, adminListUsers, adminCreateUser, adminUpdateUserRoles } from "@/providers/Auth";
import type { UserRecord } from "@/lib/auth-client";

export default function UserManagement() {
  const { token, isAdmin, user, loading: authLoading } = useAuth();
  const [users, setUsers] = useState<UserRecord[]>([]);
  const [loading, setLoading] = useState(false);
  const [creating, setCreating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [formEmail, setFormEmail] = useState("");
  const [formPassword, setFormPassword] = useState("");
  const [formFullName, setFormFullName] = useState("");
  const [formAdmin, setFormAdmin] = useState(false);

  const loadUsers = useCallback(async () => {
    if (!token || !isAdmin) return;
    setLoading(true);
    setError(null);
    try {
      const list = await adminListUsers(token);
      setUsers(list);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Failed to load users";
      setError(message);
      toast.error("Failed to load users", { description: message });
    } finally {
      setLoading(false);
    }
  }, [token, isAdmin]);

  useEffect(() => {
    void loadUsers();
  }, [loadUsers]);

  if (authLoading) {
    return (
      <div className="flex min-h-[320px] items-center justify-center text-sm text-muted-foreground">
        Checking access...
      </div>
    );
  }

  if (!token || !isAdmin) {
    return (
      <div className="flex min-h-[320px] items-center justify-center text-sm text-muted-foreground">
        Admin access required.
      </div>
    );
  }

  const handleCreate = async () => {
    if (!formEmail || !formPassword) {
      toast.error("Email and password are required");
      return;
    }
    setCreating(true);
    setError(null);
    try {
      const payloadRoles = formAdmin ? ["admin", "user"] : ["user"];
      const created = await adminCreateUser(token, {
        email: formEmail,
        password: formPassword,
        full_name: formFullName || null,
        roles: payloadRoles,
      });
      setUsers((prev) => [created, ...prev]);
      setFormEmail("");
      setFormPassword("");
      setFormFullName("");
      setFormAdmin(false);
      toast.success("User created");
    } catch (err) {
      const message = err instanceof Error ? err.message : "Failed to create user";
      setError(message);
      toast.error("Failed to create user", { description: message });
    } finally {
      setCreating(false);
    }
  };

  const handleToggleAdmin = async (target: UserRecord, nextValue: boolean) => {
    if (!token) return;
    const baseRoles = target.roles.filter((role) => role !== "admin");
    const roles = nextValue ? Array.from(new Set([...baseRoles, "admin", "user"])) : Array.from(new Set([...baseRoles, "user"]));
    try {
      const updated = await adminUpdateUserRoles(token, target.id, roles);
      setUsers((prev) => prev.map((item) => (item.id === updated.id ? updated : item)));
      toast.success(`Updated roles for ${updated.email}`);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Failed to update roles";
      toast.error("Failed to update roles", { description: message });
    }
  };

  return (
    <div className="mx-auto flex w-full max-w-4xl flex-col gap-6 p-6">
      <Card>
        <CardHeader>
          <CardTitle>Create user</CardTitle>
          <CardDescription>Provision agent users and assign roles.</CardDescription>
        </CardHeader>
        <CardContent className="grid gap-4">
          <div className="grid gap-2">
            <Label htmlFor="admin-email">Email</Label>
            <Input
              id="admin-email"
              type="email"
              autoComplete="email"
              value={formEmail}
              onChange={(event) => setFormEmail(event.target.value)}
              required
            />
          </div>
          <div className="grid gap-2">
            <Label htmlFor="admin-password">Password</Label>
            <Input
              id="admin-password"
              type="password"
              autoComplete="new-password"
              value={formPassword}
              onChange={(event) => setFormPassword(event.target.value)}
              required
            />
          </div>
          <div className="grid gap-2">
            <Label htmlFor="admin-full-name">Full name</Label>
            <Input
              id="admin-full-name"
              value={formFullName}
              onChange={(event) => setFormFullName(event.target.value)}
            />
          </div>
          <div className="flex items-center justify-between rounded-md border px-3 py-2">
            <div>
              <p className="font-medium text-sm">Administrator</p>
              <p className="text-muted-foreground text-xs">Grants access to user management and advanced controls.</p>
            </div>
            <Switch
              checked={formAdmin}
              onCheckedChange={setFormAdmin}
              disabled={creating}
            />
          </div>
          <div className="flex justify-end gap-2">
            <Button
              type="button"
              variant="outline"
              onClick={() => {
                setFormEmail("");
                setFormPassword("");
                setFormFullName("");
                setFormAdmin(false);
              }}
              disabled={creating}
            >
              Clear
            </Button>
            <Button
              type="button"
              onClick={handleCreate}
              disabled={creating}
            >
              {creating ? "Creating..." : "Create"}
            </Button>
          </div>
          {error ? (
            <p className="text-sm text-destructive" role="alert">
              {error}
            </p>
          ) : null}
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="flex flex-row items-center justify-between">
          <div>
            <CardTitle>Users</CardTitle>
            <CardDescription>Manage roles and audit access.</CardDescription>
          </div>
          <div className="flex items-center gap-2">
            <Button
              type="button"
              variant="outline"
              onClick={() => void loadUsers()}
              disabled={loading}
            >
              {loading ? "Refreshing..." : "Refresh"}
            </Button>
          </div>
        </CardHeader>
        <CardContent className="space-y-3">
          {users.length === 0 ? (
            <p className="text-sm text-muted-foreground">
              {loading ? "Loading users..." : "No users yet."}
            </p>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full min-w-[480px] border-collapse text-sm">
                <thead>
                  <tr className="border-b bg-muted/40 text-left">
                    <th className="px-3 py-2 font-medium">Email</th>
                    <th className="px-3 py-2 font-medium">Name</th>
                    <th className="px-3 py-2 font-medium">Roles</th>
                    <th className="px-3 py-2 font-medium text-right">Admin</th>
                  </tr>
                </thead>
                <tbody>
                  {users.map((item) => {
                    const isSelf = user?.id === item.id;
                    const isAdminRole = item.roles.includes("admin");
                    return (
                      <tr
                        key={item.id}
                        className="border-b last:border-b-0"
                      >
                        <td className="px-3 py-2 font-medium">{item.email}</td>
                        <td className="px-3 py-2">{item.full_name || "—"}</td>
                        <td className="px-3 py-2">{item.roles.join(", ")}</td>
                        <td className="px-3 py-2 text-right">
                          <Switch
                            checked={isAdminRole}
                            onCheckedChange={(value) => void handleToggleAdmin(item, value)}
                            disabled={loading || isSelf}
                          />
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
