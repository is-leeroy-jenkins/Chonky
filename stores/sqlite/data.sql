CREATE TABLE IF NOT EXISTS "Partitions" 
(
	"PartitionsId" INTEGER NOT NULL UNIQUE,
	"FiscalYear" TEXT(80) NOT NULL,
	"BPOA" TEXT(80) NOT NULL,
	"EPOA" TEXT(80) NOT NULL,
	"Type" TEXT(80) NOT NULL,
	"TreasuryAccountCode" TEXT(80) NOT NULL,
	"MainAccount" TEXT(80) NOT NULL,
	"BudgetAccountCode" TEXT(80) NOT NULL,
	"Amount" DOUBLE NOT NULL DEFAULT 0.0,
	"LineNumber" TEXT(80) NOT NULL,
	"LineName" TEXT(80) NOT NULL,
	PRIMARY KEY("PartitionsId" AUTOINCREMENT)
);

CREATE TABLE TABLE IF NOT EXISTS "Prompts"
(
	"PromptsId" INTEGER NOT NULL UNIQUE,
	"Name" TEXT NOT NULL,
	"Text" TEXT NOT NULL,
	"ID" TEXT,
	PRIMARY KEY("PromptsId" AUTOINCREMENT)
);

CREATE TABLE TABLE IF NOT EXISTS "Search" 
(
	"SearchId" INTEGER NOT NULL UNIQUE,
	"ID" TEXT NOT NULL,
	"Name" TEXT,
	"Location" TEXT,
	"Code" TEXT,
	PRIMARY KEY("SearchId" AUTOINCREMENT)
);

CREATE TABLE TABLE IF NOT EXISTS "Appropriations" 
(
	"AppropriationsId" INTEGER NOT NULL UNIQUE,
	"FiscalYear" TEXT(80) DEFAULT 'NS',
	"PublicLaw" TEXT(80) DEFAULT 'NS',
	"AppropriationTitle" TEXT(80) DEFAULT 'NOT SPECIFIED',
	"EnactedDate" TEXT(80) DEFAULT 'NS',
	"ExplanatoryComments" TEXT(80) DEFAULT 'NOT SPECIFIED',
	"Authority" DOUBLE DEFAULT 0.0,
	PRIMARY KEY("AppropriationsId" AUTOINCREMENT)
);

CREATE TABLE TABLE IF NOT EXISTS "Apportionments"
(
	"ApportionmentsId" INTEGER,
	"FiscalYear" TEXT(80),
	"BPOA" TEXT(80),
	"EPOA" TEXT(80),
	"MainAccount" TEXT(80),
	"TreasuryAccountCode" TEXT(80),
	"TreasuryAccountName" TEXT(80),
	"AvailabilityType" TEXT(80),
	"BudgetAccountCode" TEXT(80),
	"BudgetAccountName" TEXT(80),
	"LineNumber" TEXT(80),
	"LineSplit" TEXT(80),
	"LineName" TEXT(80),
	"Amount" DOUBLE DEFAULT 0.0,
	PRIMARY KEY("ApportionmentsId" AUTOINCREMENT)
);

CREATE TABLE IF NOT EXISTS "AgencyAccounts" 
(
	"AgencyAccountsId"	INTEGER NOT NULL UNIQUE,
	"TreasuryAgencyCode"	TEXT(80),
	"AgencyCode"	TEXT(80),
	"AgencyName"	TEXT(80),
	"BureauCode"	TEXT(80),
	"BureauName"	TEXT(80),
	"AccountCode"	TEXT(80),
	"AccountName"	TEXT(80),
	PRIMARY KEY("AgencyAccountsId" AUTOINCREMENT)
);

CREATE TABLE IF NOT EXISTS "Code of Federal Regulations Title 31 Money And Finance" 
(
	"Index"	INTEGER NOT NULL UNIQUE,
	"Question"	TEXT,
	"Answer"	TEXT,
	PRIMARY KEY("Index" AUTOINCREMENT)
);

CREATE TABLE IF NOT EXISTS "OMB Circular A-11 Preparation Submission And Execution Of The Budget" 
(
	"Index"	INTEGER NOT NULL,
	"Question"	TEXT,
	"Answer"	TEXT,
	PRIMARY KEY("Index" AUTOINCREMENT)
);

CREATE TABLE IF NOT EXISTS "OMB Circular A-11 SF-132" 
(
	"Index"	INTEGER NOT NULL UNIQUE,
	"Question"	TEXT,
	"Answer"	TEXT,
	PRIMARY KEY("Index" AUTOINCREMENT)
);

CREATE TABLE IF NOT EXISTS "OMB Circular A-11 Section 120 Apportionment Process" 
(
	"Index"	INTEGER NOT NULL UNIQUE,
	"Question"	TEXT,
	"Answer"	TEXT,
	PRIMARY KEY("Index" AUTOINCREMENT)
);

CREATE TABLE IF NOT EXISTS "Principles Of Federal Appropriations Law" 
(
	"Index"	INTEGER NOT NULL UNIQUE,
	"Question"	TEXT,
	"Answer"	TEXT,
	PRIMARY KEY("Index" AUTOINCREMENT)
);