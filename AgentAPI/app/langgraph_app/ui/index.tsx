import React from "react";

interface KnowledgeSearchStatusProps {
  operation: string;
  query?: string;
  status: "searching" | "complete" | "error";
  timestamp: string;
}

const KnowledgeSearchStatus: React.FC<KnowledgeSearchStatusProps> = ({
  operation,
  query,
  status,
}) => {
  const getIcon = () => {
    if (status === "complete") {
      return (
        <svg
          className="w-4 h-4 text-green-500"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M5 13l4 4L19 7"
          />
        </svg>
      );
    }
    if (status === "error") {
      return (
        <svg
          className="w-4 h-4 text-red-500"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M6 18L18 6M6 6l12 12"
          />
        </svg>
      );
    }
    return (
      <div className="animate-spin h-4 w-4 border-2 border-blue-500 border-t-transparent rounded-full" />
    );
  };

  const getTextColor = () => {
    if (status === "complete") return "text-green-600";
    if (status === "error") return "text-red-600";
    return "text-blue-600";
  };

  return (
    <div className="flex items-center gap-2 py-2 px-3 bg-gray-50 dark:bg-gray-800 rounded-md text-sm">
      <div className="flex-shrink-0">{getIcon()}</div>
      <div className="flex-1 min-w-0">
        <div className={`font-medium ${getTextColor()}`}>{operation}</div>
        {query && (
          <div className="text-xs text-gray-500 dark:text-gray-400 truncate mt-0.5">
            {query}
          </div>
        )}
      </div>
    </div>
  );
};

export default {
  knowledge_search_status: KnowledgeSearchStatus,
};
