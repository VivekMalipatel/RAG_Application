import { Suspense } from "react";
import UserManagement from "@/components/admin/UserManagement";
import { Skeleton } from "@/components/ui/skeleton";

export default function AdminPage() {
  return (
    <div className="flex min-h-screen flex-col bg-background">
      <header className="border-b">
        <div className="mx-auto w-full max-w-5xl px-6 py-8">
          <h1 className="text-2xl font-semibold tracking-tight">Admin console</h1>
          <p className="text-sm text-muted-foreground">
            Manage workspace members, create accounts, and grant administrator access.
          </p>
        </div>
      </header>
      <main className="flex-1">
        <Suspense
          fallback={(
            <div className="mx-auto w-full max-w-4xl p-6">
              <Skeleton className="h-[400px] w-full" />
            </div>
          )}
        >
          <UserManagement />
        </Suspense>
      </main>
    </div>
  );
}
